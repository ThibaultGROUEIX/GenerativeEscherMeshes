import os
import pickle
import time

import igl
import numpy as np
import PIL
import torch
import torchvision
from rich.pretty import pprint
from rich.traceback import install
from tqdm import tqdm

import nvdiffrast.torch as dr

install()
from pathlib import Path
import torch.nn.functional as F
from omegaconf import OmegaConf

import escher.rendering.renderer_nvdiffrast as renderer
import escher.guidance.sd as sd
import escher.geometry.split_square_boundary as split_square_boundary
from escher.misc.color_conversion import hsv2rgb_torch
from escher.geometry.get_base_mesh import get_2d_square_mesh, get_hexagonal_mesh
from escher.geometry.vertex_augmentation import vertex_augmentation
from escher.geometry.equal_area_loss import EqualAreaLoss
from escher.rendering.render_mesh_from_path import render_from_path
from escher.rendering.render_tiling_core import render_tiling
from escher.rendering.render_mesh_matplotlib import render_mesh_matplotlib
from escher.geometry.save_mesh import save_mesh
from escher.geometry.sanity_checks import check_triangle_orientation
from escher.OTE.core import OTESolver
from escher.OTE.tilings import (
    Cylinder,
    KleinBottle,
    MobiusStrip,
    OrbifoldI,
    OrbifoldIHybrid,
    OrbifoldII,
    OrbifoldII_hexagon,
    OrbifoldIIHybrid,
    OrbifoldIII,
    OrbifoldIV,
    OrbifoldIVHybrid,
    PinnedBoundary,
    ProjectivePlane,
    ReflectSquare,
    Reflect442,
    Reflect333,
    Reflect632,
    RightAngleHybrid,
    Torus,
    Torus_hexagon
)
from escher.misc.misc import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, seed_everything
from escher.OTE.core import GlobalDeformation

start_time = time.time()

# get path to this file
PATH = Path(__file__).parent.absolute()


class Escher:
    def __init__(self) -> None:
        # args = parser.parse_args()
        cli_conf = OmegaConf.from_cli()
        conf_file = cli_conf.get("CONF_FILE", "configs/base.yaml")
        base_conf = OmegaConf.load(PATH / f"{conf_file}")
        args = OmegaConf.merge(base_conf, cli_conf)
        seed_everything(args.SEED)

        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        with open(f"{args.OUTPUT_DIR}/config.yaml", "w") as fp:
            OmegaConf.save(config=args, f=fp.name)
        pprint(dict(args))

        # ================== Parameter processing ===========================
        # self.N_NO_TEXTURE_IMAGES = int(args.NO_TEXTURE_IMAGES_IN_BATCH_PERCENTAGE * args.IMAGE_BATCH_SIZE / 100.0)
        # self.N_TEXTURE_IMAGES = args.IMAGE_BATCH_SIZE - self.N_NO_TEXTURE_IMAGES
        self.args = args
        self.intermediate_folder = os.path.join(args.OUTPUT_DIR, "intermediate")
        self.device = torch.device(args.DEVICE)
        if args.NO_DEFORMATION:
            DEVICE = "cuda:0"

        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.intermediate_folder, exist_ok=True)

        self.init_renderer()
        if self.args.EXPERIMENT != "OUTLINE":
            self.init_guidance()
            print("Guidance model loaded")
            self.init_backgrounds()
            print("Backgrounds loaded")
            self.get_embeddings()
            print("Embeddings loaded")

        self.init_mesh_and_solver()
        print("Mesh and solver loaded")
        self.init_optimizer()
        print("Optimizer loaded")
        self.init_loss()
        print("Loss loaded")
        if self.args.SYMMETRY_EXPERIMENT:
            self.init_symmetry_experiment()

    def init_symmetry_experiment(self):
        # Repeat the base vertex two times.

        self.uv = torch.cat([self.uv] * self.constraint_data.symmetric_copies, dim=1)
        new_faces = self.faces + len(self.points)
        if self.args.TILING_TYPE in ["Cylinder", "MobiusStrip", "RightAngleHybrid", "OrbifoldIHybrid", "OrbifoldIIHybrid", "OrbifoldIVHybrid"]:
            # Permute the order of the triangles to keep all triangles correctly oriented after reflection. We test for this in the main training loop
            new_faces = new_faces[:, [0, 2, 1]]
            self.faces = torch.cat([self.faces, new_faces], dim=0)
            self.faces_split = [torch.cat([face, face + len(self.points)], dim=0) for face in self.faces_split]
        else:
            # There might be more than one copy in the rotational cases (this is arbitrary)
            self.faces = torch.cat([self.faces + i * len(self.points)  for i in range(self.constraint_data.symmetric_copies)], dim=0)
            self.faces_split = [
                torch.cat([face + i * len(self.points) for i in range(self.constraint_data.symmetric_copies)], dim=0)
                for face in self.faces_split
            ]

        self.faces_npy = self.faces.cpu().numpy()

    def init_renderer(self):
        # ================== Init Renderer ===========================
        self.glctx = dr.RasterizeCudaContext()
        self.mv = torch.eye(4, device=self.device)[None, ...]
        self.proj = torch.eye(4, device=self.device)[None, ...]

    def init_guidance(self):
        if self.args.GUIDANCE == "SD":
            # ================== Init Stable Diffusion ===========================
            textual_inversion_path = ""
            if self.args.TEXTUAL_INVERSION:
                textual_inversion_path = "/home/groueix/GenerativeEscherPatterns/diffusers/examples/textual_inversion/textual_inversion_cat/learned_embeds.safetensors"

            config = sd.Config(
                pretrained_model_name_or_path = self.args.PRETRAINED_MODEL_NAME_OR_PATH,
                textual_inversion=textual_inversion_path,
                guidance_scale=self.args.GUIDANCE_SCALE,
                half_precision_weights=self.args.USE_HALF_PRECISION,
                grad_clip=[0, 2.0, 8.0, 1000] if self.args.CLIP_GRADIENTS_IN_SDS else None,
            )
            pprint(config)
            self.guidance_model = sd.StableDiffusion(config)
        elif self.args.GUIDANCE == "IF":
            # sys.path.append("/home/groueix/tiling/threestudio")
            import escher.guidance.deepfloyd as deep_floyd_guidance

            self.guidance_model = deep_floyd_guidance.DeepFloydGuidance()
            # self.guidance_model.configure()
        else:
            print(
                f"No guidance specified or unsuported guidance. Guidance is {self.args.GUIDANCE}, and should be either of SD or IF"
            )

    def get_embeddings(self):
        if isinstance(self.args.PROMPT, str):
            self.args.PROMPT = [self.args.PROMPT]

        with torch.no_grad():
            negative_embedding = self.guidance_model.get_text_embeds(self.args.NEGATIVE_PROMPT)
            prompt_embedding = [self.guidance_model.get_text_embeds(prompt) for prompt in self.args.PROMPT]
            n_prompt = len(self.args.PROMPT)

            if n_prompt == 1:
                self.text_embeds = torch.cat(
                    prompt_embedding * self.args.IMAGE_BATCH_SIZE + [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                )
            elif n_prompt == 2:
                # The text batch is organized as follows
                # [prompt1, ..., prompt1 |  prompt2, ..., prompt2 |  negative, ..., negative |  negative, ..., negative]
                # the corresponding rendering batch is organized as follows
                # [mesh1, ..., mesh1 |  mesh2, ..., mesh2]
                assert self.args.IMAGE_BATCH_SIZE % 2 == 0, "Batch size must be even"
                batch_size_per_prompt = self.args.IMAGE_BATCH_SIZE // 2
                positive_embedding = [prompt_embedding[0]] * batch_size_per_prompt + [
                    prompt_embedding[1]
                ] * batch_size_per_prompt
                negative_embedding = [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                self.text_embeds = torch.cat(positive_embedding + negative_embedding)
                self.batch_size_per_prompt = batch_size_per_prompt
            else:
                assert (np.sqrt(n_prompt) % 1 == 0), "Number of prompts must be a square number"
                # The text batch is organized as follows
                # [prompt1, ..., prompt1 |  prompt2, ..., prompt2 | ... | prompt_k,..., prompt_k | negative, ..., negative |  negative, ..., negative |  negative, ..., negative]
                # the corresponding rendering batch is organized as follows
                # [mesh1, ..., mesh1 |  mesh2, ..., mesh2 | ... |  meshk, ..., meshk]
                assert self.args.IMAGE_BATCH_SIZE % n_prompt == 0, "Batch size must be a multiple of n_prompt"
                batch_size_per_prompt = self.args.IMAGE_BATCH_SIZE // n_prompt
                positive_embedding = []
                for i in range(n_prompt):
                    positive_embedding += [prompt_embedding[i]] * batch_size_per_prompt
                negative_embedding = [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                self.text_embeds = torch.cat(positive_embedding + negative_embedding)
                self.batch_size_per_prompt = batch_size_per_prompt

            del self.guidance_model.text_encoder

    def init_backgrounds(self):
        self.black_bg = torch.zeros(1, 3, self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM).cuda()
        self.white_bg = torch.ones(1, 3, self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM)

    def constraints_from_args(self, vertices, sides):
        """Returns the constraint data object corresponding to the tiling type specified in the args"""

        if self.args.TILING_TYPE == "Cylinder":
            return Cylinder.CylinderConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "KleinBottle":
            return KleinBottle.KleinBottleConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "MobiusStrip":
            return MobiusStrip.MobiusStripConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldI":
            return OrbifoldI.OrbifoldIConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldIHybrid":
            return OrbifoldIHybrid.OrbifoldIHybridConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldII":
            return OrbifoldII.OrbifoldIIConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldIIHexagon":
            return OrbifoldII_hexagon.OrbifoldIIHexagonConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldIIHybrid":
            return OrbifoldIIHybrid.OrbifoldIIHybridConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldIII":
            return OrbifoldIII.OrbifoldIIIConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldIV":
            return OrbifoldIV.OrbifoldIVConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "OrbifoldIVHybrid":
            return OrbifoldIVHybrid.OrbifoldIVHybridConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "PinnedBoundary":
            return PinnedBoundary.PinnedBoundaryConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "ProjectivePlane":
            return ProjectivePlane.ProjectivePlaneConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "ReflectSquare":
            return ReflectSquare.ReflectSquareConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "Reflect442":
            return Reflect442.Reflect442Constraints(vertices, sides)

        elif self.args.TILING_TYPE == "Reflect333":
            return Reflect333.Reflect333Constraints(vertices, sides)

        elif self.args.TILING_TYPE == "Reflect632":
            return Reflect632.Reflect632Constraints(vertices, sides)

        elif self.args.TILING_TYPE == "RightAngleHybrid":
            return RightAngleHybrid.RightAngleHybridConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "Torus":
            return Torus.TorusConstraints(vertices, sides)

        elif self.args.TILING_TYPE == "TorusHexagon":
            return Torus_hexagon.TorusHexagonConstraints(vertices, sides)

        else:
            print(
                "The only valid tiling tiles are : Cylinder, KleinBottle, MobiusStrip, OrbifoldI, OrbifoldIHybrid, OrbifoldII,OrbifoldIIHybrid, OrbifoldIII, OrbifoldIV,OrbifoldIVHybrid, PinnedBoundary, ProjectivePlane, ReflectSquare, RightAngleHybrid, Torus"
            )
            print("You provided : ", self.args.TILING_TYPE)
            raise Exception("unknown tiling type")

    def init_mesh_and_solver(self):
        # ============== generate a 2D mesh of a square =======================
        assert self.args.MESH_RESOLUTION % 2 == 0, "mesh resolution must be even for some wallpaper groups"
        assert (np.sqrt(len(self.args.PROMPT)) % 1 == 0) or len(
            self.args.PROMPT
        ) == 2, "number of prompts must be a square number, or 2"

        points, faces_npy, faces_split, mask = get_2d_square_mesh(
            self.args.MESH_RESOLUTION, num_labels=len(self.args.PROMPT)
        )
        if "Hexagon" in self.args.TILING_TYPE:
            # Overwrite with hexagon
            assert len(self.args.PROMPT) == 1, "Hexagon experiment only works with one prompt"
            points, faces_npy, sides = get_hexagonal_mesh(vertices_per_edge = self.args.MESH_RESOLUTION)
            faces_split = [faces_npy]

        # =========== init uv ==================================================
        normalized_points = points
        normalized_points = normalized_points - normalized_points.min()
        uv = normalized_points / normalized_points.max() # [0,1]
        uv = torch.from_numpy(uv).unsqueeze(0).to(self.device)
        normalized_points = 2 * normalized_points / normalized_points.max() - 1 # [-1,1]

        # tri = Delaunay(points)
        faces = torch.from_numpy(faces_npy)

        # bdry indices of mesh
        bdry = igl.boundary_loop(faces_npy)

        # split the bdry into 4 sides (left,right,top,down)
        if not ("Hexagon" in self.args.TILING_TYPE):
            sides = split_square_boundary.split_square_boundary(points, bdry)

        # generate nx2 list of edge pairs (i,j)
        adjacency_list = igl.adjacency_list(faces_npy)  # list of lists containing at index i the adjacent vertices of vertex i

        edge_pairs = []
        for r, i in zip(adjacency_list, range(len(adjacency_list))):
            for j in r:
                if i < j:
                    edge_pairs.append((i, j))
        edge_pairs = np.asarray(edge_pairs)

        constraint_data = self.constraints_from_args(points, sides)

        # the solver itself
        solver = OTESolver.OTESolver(edge_pairs, points, constraint_data)
        W = torch.nn.Parameter(torch.randn((edge_pairs.shape[0], 1)))
        self.faces_split = [
            torch.from_numpy(faces_split_).to(self.device).type(torch.int32) for faces_split_ in faces_split
        ]

        self.points = points
        self.uv = uv
        self.bdry = bdry
        self.faces = faces
        self.faces_npy = faces_npy
        self.constraint_data = constraint_data
        # self.ROTATION_MATRIX = ROTATION_MATRIX
        self.global_map = GlobalDeformation.GlobalDeformation(
            constraint_data.get_horizontal_symmetry_orientation(), device=self.device
        )
        # self.global_map.to(self.device)
        self.solver = solver
        self.W = W
        self.sides = sides
        self.edge_pairs = edge_pairs

    def init_optimizer(self):
        # ================== Init Texture ===========================
        texture = None
        self.num_channels = 3

        if self.args.OPTIMISE_TEXTURE_MAP:
            texture_map_resolution = max(1024, 512 * len(self.args.PROMPT))
            self.color_parameters = torch.nn.Parameter(
                torch.rand(1, texture_map_resolution, texture_map_resolution, self.num_channels).to(self.device)
            )
        else:
            self.color_parameters = torch.nn.Parameter(torch.rand(self.W.shape[0], self.num_channels))

        # global_A = torch.nn.Parameter(torch.eye(2,device='cuda:0'))
        # self.global_theta = torch.nn.Parameter(torch.zeros(1, device="cuda:0"))
        # self.global_sym_ab = torch.nn.Parameter(torch.Tensor([1, 0]).to(self.device))
        # global_theta2 = torch.nn.Parameter(torch.zeros(1,device='cuda:0'))
        if self.args.NO_DEFORMATION:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.color_parameters, "lr": self.args.LR_COLOR},
                ],
                lr=self.args.LR,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.W, "lr": self.args.LR},
                    {"params": self.global_map.theta1, "lr": self.args.LR / 20},
                    {"params": self.global_map.theta2, "lr": self.args.LR / 20},
                    {"params": self.global_map.singular_value, "lr": self.args.LR / 40},
                    {"params": self.color_parameters, "lr": self.args.LR_COLOR},
                ],
                lr=self.args.LR,
            )
            self.init_scheduler()

    def init_scheduler(self, remaining_steps=None):
        if remaining_steps is None:
            remaining_steps = self.args.N_STEPS
        if self.args.SCHEDULER == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 100, int(self.self.args.N_STEPS * 1.5))
        elif self.args.SCHEDULER == "linear":
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 100)
        elif self.args.SCHEDULER == "step":
            # Decrease LR by 10 at 80% of the steps
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, remaining_steps - int(0.2 * remaining_steps), gamma=0.1, last_epoch=-1, verbose=False
            )
        else:
            print(f"Scheduler should be either cosine, linear or step. You provided {self.args.SCHEDULER}")
            raise Exception("Unknown scheduler")

    def init_loss(self):
        # ================== Init Loss ===========================
        if self.args.USE_TARGET_IMAGE:
            target_image = torch.from_numpy(np.asarray(PIL.Image.open("./dummy_target.png"))).cuda() / 255
            target_image = torch.einsum("hwc->chw", target_image)
            target_image = F.interpolate(
                target_image.unsqueeze(0), (self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM), mode="nearest"
            )
            self.target_image = torch.einsum("bchw->bhwc", target_image).cuda()
        if len(self.args.PROMPT) > 1:
            self.area_loss = EqualAreaLoss()

    def render(self, batch_vertices, colors_, texture_batched):
        rendered_img = None
        if len(self.args.PROMPT) == 1:

            rendered_img = renderer.render_mesh_nvdiffrast(
                vertices=batch_vertices.float(),
                faces=self.faces,
                vertices_color=colors_,
                uv=self.uv,
                mv=self.mv,
                proj=self.proj,
                image_size=(self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM),
                glctx=self.glctx,
                texture=texture_batched,
            )[0]
        else:
            img_list = []
            for i in range(len(self.args.PROMPT)):
                img_list.append(
                    renderer.render_mesh_nvdiffrast(
                        vertices=batch_vertices.float()[
                            (i) * self.batch_size_per_prompt : (i + 1) * self.batch_size_per_prompt
                        ],
                        faces=self.faces_split[i],
                        vertices_color=colors_,
                        uv=self.uv,
                        mv=self.mv,
                        proj=self.proj,
                        image_size=(self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM),
                        glctx=self.glctx,
                        texture=texture_batched[
                            (i) * self.batch_size_per_prompt : (i + 1) * self.batch_size_per_prompt
                        ],
                    )[0]
                )
            rendered_img = torch.cat(img_list, dim=0)

        # ================== Post-process the background to be random ===========================
        if self.args.RANDOM_BACKGROUND:
            mask = rendered_img[:, :, :, self.num_channels :].clone()
            # Replace background by a random color
            bg = torch.rand_like(rendered_img)[:, :1, :1, :].expand(rendered_img.shape).contiguous()
            rendered_img = mask * rendered_img + (1 - mask) * bg
            rendered_img[:, :, :, self.num_channels :] = mask

        # ================== make a tight crop around the object ===========================
        if self.args.CROP_RENDERINGS:
            try:
                from escher.rendering.crop_rendering import crop

                rendered_img = crop(rendered_img)
            except:
                print("Cropping failed")
                torchvision.utils.save_image(
                    rendered_img.permute(0, 3, 1, 2), os.path.join(self.args.OUTPUT_DIR, f"rendered__cropbug.png")
                )
        return rendered_img

    def run(self):
        # output pbar to stdout
        pbar = tqdm(total=self.args.N_STEPS, desc="steps", position=0)

        # ================== Main training loop ===========================
        for iter in range(self.args.N_STEPS):
            if iter == self.args.ONLY_TEXTURE_FROM_THIS_POINT:
                self.optimizer = torch.optim.Adam(
                    [
                        {"params": self.color_parameters, "lr": self.args.LR_COLOR},
                    ],
                    lr=self.args.LR,
                )
                self.init_scheduler(remaining_steps=self.args.N_STEPS - iter)
                mapped = mapped.clone().detach().requires_grad_(False)

            if iter < self.args.ONLY_TEXTURE_FROM_THIS_POINT:
                if self.args.CLAMP_TEXTURE:
                    self.color_parameters.data = self.color_parameters.data.clip(0, 1)

                self.optimizer.zero_grad()
                # weights are positive and smaller than 1
                if self.args.SIGMOID_WEIGHTS:
                    # self.W.data = self.W.data.clip(-10, 10) #after 10 we're praxtically at 1 for sigmoid This is killing gradients
                    w = torch.special.expit(self.W)
                else:
                    self.W.data = self.W.data.clip(0, 1)
                    w = self.W
                w_solver_input = w * self.args.W_RANGE + (1 - self.args.W_RANGE) / 2
                # [0,1] ----> [r, 1-r] where r = (1-W_RANGE)/2

                # ======Solve linear solve ==========================================================
                mapped, _, success = self.solver.solve(w_solver_input)
                if not success:
                    # pickle everything i need to reproduce
                    os.makedirs(os.path.join(self.args.OUTPUT_DIR, "failure"), exist_ok=True)
                    with open(os.path.join(self.args.OUTPUT_DIR, "failure", f"results_{iter}.pkl"), "wb") as f:
                        pickle.dump(
                            {
                                "V": mapped.detach().cpu().numpy(),
                                "T": self.faces.cpu().numpy(),
                                "UV": self.uv[0, ...].detach().cpu().numpy(),
                                "bdry": self.bdry,
                                "constraint_data": self.constraint_data,
                                "w_solver_input": w_solver_input,
                                "config": dict(self.args),
                                "faces_split": [faces_split_.cpu().numpy() for faces_split_ in self.faces_split],
                            },
                            f,
                        )
                mapped = mapped.cuda().float() 
                # ====================================================================================

                # ====== Repeat the tile one time if symmetry experiment is enabled =====================
                if self.args.SYMMETRY_EXPERIMENT:
                    # Repeat the base vertex two times.
                    symmetry_maps = self.constraint_data.get_symmetry_map(vertices=mapped, sides=self.constraint_data.sides)
                    mapped_copies = [map.map(mapped[:, 0:2]) for map in symmetry_maps]
                    mapped = torch.cat(mapped_copies, dim=0)
                # ====================================================

                # ====== More sanity check on self overlaps =====================
                try:
                    check_triangle_orientation(mapped, self.faces)
                except:
                    # Dump info in global failure folder, append to existing
                    with open("failure.txt", "a") as f:
                        f.write(f"Failure at iter {iter}")
                        # dump config in txt file for reproducibility
                        f.write(f"Config: {self.args}")
                # ====================================================

                mapped_noglobaltransform = mapped
                global_A = torch.eye(2).cuda()
                if self.args.GLOBAL_AFFINE:
                    # TODO thibault - add arg to control whether want global_rotation or not -- I think we always do
                    global_A = self.global_map.get_matrix(
                        global_rotation=True, map_type=self.constraint_data.get_global_transformation_type()
                    )
                    mapped = GlobalDeformation.map(mapped, global_A)

            if self.args.BW:
                texture = torch.tile(self.color_parameters[:, :, :, :1], (1, 1, 1, 3))
            else:
                texture = self.color_parameters

            if self.args.CLAMP_TEXTURE:
                pass
            else:
                texture = torch.sigmoid(texture)

            if not self.args.OPTIMISE_TEXTURE_MAP:
                texture = None

            if self.args.USE_HSV:
                # temp = torch.concat([torch.zeros([texture.shape[1],texture.shape[2],2]).cuda(),texture[0,:,:,:1]],axis=2).unsqueeze(0)
                # texture = hsv2rgb_torch(temp.transpose(3,1)).transpose(3,1)
                texture = hsv2rgb_torch(texture.transpose(3, 1)).transpose(3, 1)
            # otherwise, it is set to None and we use per-vertex colors

            colors_ = (
                torch.sigmoid(self.color_parameters) if not self.args.OPTIMISE_TEXTURE_MAP else None
            )  # *0.5+0.5 #colors.clip(0,1)

            # ================== Drop the texture a percentage of the batch ===========================
            drop = np.random.choice(
                [0, 1],
                size=self.args.IMAGE_BATCH_SIZE,
                p=[1 - self.args.TEXTURE_DROP_PROB / 100.0, self.args.TEXTURE_DROP_PROB / 100.0],
            )
            drop = drop.astype("bool")
            drop[0] = False  # Small hack for visualization purposes
            texture_batched = texture.repeat(self.args.IMAGE_BATCH_SIZE, 1, 1, 1).contiguous()
            texture_batched[drop] = (
                texture_batched[drop] * 0 + torch.rand(self.args.IMAGE_BATCH_SIZE, 1, 1, 1, device=texture.device)[drop]
            )

            # ================== Randomly augment the vertices ===========================

            batch_vertices = vertex_augmentation(
                mapped,
                ROTATION_MATRIX=torch.eye(2).to(self.device),
                RANDOM_RIGID=self.args.RANDOM_RIGID,
                NO_DEFORMATION=self.args.NO_DEFORMATION,
                IMAGE_BATCH_SIZE=self.args.IMAGE_BATCH_SIZE,
            )

            # ================== Render the mesh ===========================
            rendered_img = self.render(batch_vertices, colors_, texture_batched)

            # ================== Compute loss ===========================
            if self.args.USE_TARGET_IMAGE:
                loss = torch.sum((rendered_img.squeeze()[:, :, : self.num_channels] - self.target_image) ** 2)
            else:
                loss, t = self.guidance_model.train_step(
                    rgb=rendered_img[:, :, :, : self.num_channels].contiguous(), text_embeddings=self.text_embeds
                )
            loss = loss + self.args.W_REGULARIZATION * torch.sum(self.W**2)
            if len(self.args.PROMPT) > 1:
                loss = loss + self.args.AREA_REGULARIZATION * self.area_loss.equal_area_loss(
                    V=mapped, faces_split=self.faces_split
                )

            # mid_point = int(self.sides[0].shape[0] / 2)
            # loss += torch.sum((mapped_noglobaltransform[self.sides[0][mid_point], :] - 40)**2) * 500
            # loss += torch.sum((mapped_noglobaltransform[self.edge_pairs[:,0],:] - mapped_noglobaltransform[self.edge_pairs[:,1],:])**2)*50

            # loss = loss + torch.norm(global_Sym/2,'nuc')
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # ================== Progress bar and logging ===========================
            # update the progress bar with per-iteration information
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
            self.visualizer(iter, mapped_noglobaltransform, texture, global_A, batch_vertices, rendered_img, w_solver_input)

            # Spatially varying UVs instead of fixed UVs
            if self.args.UV == "SPATIAL":
                normalized_points = mapped_noglobaltransform - mapped_noglobaltransform.min()
                self.uv = uv = (normalized_points / normalized_points.max()).unsqueeze(0).detach()

        # Save final assets
        self.visualizer(
            iter,
            mapped_noglobaltransform,
            texture,
            global_A,
            batch_vertices,
            rendered_img,
            w_solver_input,
            save_anyway=True,
            make_videos=True,
        )

    def visualizer(
        self,
        iter,
        mapped_noglobaltransform,
        texture,
        global_A,
        batch_vertices,
        rendered_img,
        w_solver_input,
        save_anyway=False,
        make_videos=False,
    ):
        texture_img = texture.detach().cpu().squeeze().permute(2, 0, 1)
        if (self.args.TILING_RENDER_FREQ > 0 and iter % self.args.TILING_RENDER_FREQ == 0) or save_anyway:
            with open(os.path.join(self.args.OUTPUT_DIR, f"results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "V": mapped_noglobaltransform.detach().cpu().numpy(),
                        "T": self.faces.cpu().numpy(),
                        "UV": self.uv[0, ...].detach().cpu().numpy(),
                        "texture": texture_img.cpu().detach().numpy(),
                        "bdry": self.bdry,
                        "constraint_data": self.constraint_data,
                        "R": global_A.cpu().detach().numpy(),
                        "w_solver_input": w_solver_input,
                        "config": dict(self.args),
                        "faces_split": [faces_split_.cpu().numpy() for faces_split_ in self.faces_split],
                    },
                    f,
                )
        # Visualizations
        step = iter // self.args.TILING_RENDER_FREQ
        # format as three digits
        step = f"{step:04d}"
        if (self.args.TILING_RENDER_FREQ > 0 and iter % self.args.TILING_RENDER_FREQ == 0) or save_anyway:
            render_tiling(
                mapped_noglobaltransform.cuda(),
                self.uv[0, ...],
                self.faces,
                texture[0, ...],
                self.bdry,
                self.constraint_data,
                global_A,  # torch.matmul(self.ROTATION_MATRIX, global_A),
                [
                    os.path.join(self.intermediate_folder, f"{step}_tile.png"),
                    os.path.join(self.args.OUTPUT_DIR, f"tile.png"),
                ],
                grid_sizes=[5],
                color_strategy="PERIODIC",
                make_video=True if save_anyway else False,
                num_labels=len(self.args.PROMPT),
                faces_split=self.faces_split,
                highlight_single_tile=True if save_anyway else False,
            )
        if save_anyway:
            # get more tilings
            render_tiling(
                mapped_noglobaltransform.cuda(),
                self.uv[0, ...],
                self.faces,
                texture[0, ...],
                self.bdry,
                self.constraint_data,
                global_A,  # torch.matmul(self.ROTATION_MATRIX, global_A),
                [
                    os.path.join(self.intermediate_folder, f"{step}_tile.png"),
                    os.path.join(self.args.OUTPUT_DIR, f"tile.png"),
                ],
                grid_sizes=[8, 20, 50, 100],
                color_strategy="PERIODIC",
                make_video=False,
                num_labels=len(self.args.PROMPT),
                faces_split=self.faces_split,
                highlight_single_tile=False,
            )
            # This one just to get the infinite video
            try:
                render_tiling(
                    mapped_noglobaltransform.cuda(),
                    self.uv[0, ...],
                    self.faces,
                    texture[0, ...],
                    self.bdry,
                    self.constraint_data,
                    global_A,  # torch.matmul(self.ROTATION_MATRIX, global_A),
                    [
                        os.path.join(self.intermediate_folder, f"{step}_tile.png"),
                        os.path.join(self.args.OUTPUT_DIR, f"tile.png"),
                    ],
                    grid_sizes=[15],
                    color_strategy="PERIODIC",
                    make_video=False,
                    num_labels=len(self.args.PROMPT),
                    faces_split=self.faces_split,
                    highlight_single_tile=False,
                    make_infinite_video=True,
                )
            except:
                print("Failed to render infinite video")

        if (iter % self.args.VISUALIZATION_FREQ == 0) or save_anyway:
            if len(self.args.PROMPT) > 1:
                self.area_loss.save_curves(path=os.path.join(self.args.OUTPUT_DIR, f"area_loss.png"))
            with torch.no_grad():
                # ============ Log mesh ============
                if not self.args.NO_DEFORMATION:
                    render_mesh_matplotlib(
                        vertices_npy=batch_vertices[0, ...].cpu().detach().numpy(),
                        faces=self.faces_npy,
                        sides=self.sides,
                        fnames=[
                            os.path.join(self.intermediate_folder, f"{step}_mesh.png"),
                            os.path.join(self.args.OUTPUT_DIR, f"mesh.png"),
                        ],
                    )

                # ============ Log render ============
                img = rendered_img[:, :, :, : self.num_channels].permute(0, 3, 1, 2)
                img = img.detach().cpu()

                torchvision.utils.save_image(img, os.path.join(self.intermediate_folder, f"{step}_render.png"))
                torchvision.utils.save_image(img[0:1], os.path.join(self.args.OUTPUT_DIR, f"last_result.png"))
                img_log = (img[0:1] * 255).floor().clip(0, 255)
                percentage = ((img[0:1] > 1.0).sum() + (img[0:1] < 0.0).sum()) / (img[0:1] * 0 + 1).sum()
                print(f"max {img[0:1].max()}, min {img[0:1].min()} Percentage of out of range pixel {percentage*100}")
                PIL.Image.fromarray(img_log.squeeze().permute(1, 2, 0).numpy().astype("uint8")).save(
                    os.path.join(self.intermediate_folder, f"last_result_true_colors.png")
                )

                # ============ Log texture map ============
                if self.args.OPTIMISE_TEXTURE_MAP:
                    texture_img = texture.detach().cpu().squeeze().permute(2, 0, 1)
                    torchvision.utils.save_image(
                        texture_img, os.path.join(self.intermediate_folder, f"{step}_texture.png")
                    )

                # R = torch.matmul(self.ROTATION_MATRIX, global_A).cpu().detach().cpu()
                # global_A
                vertex_save = GlobalDeformation.map(mapped_noglobaltransform.detach().cpu(), global_A)
                save_mesh(
                    os.path.join(self.args.OUTPUT_DIR, f"final_mesh.obj"),
                    np.concatenate((vertex_save, np.zeros((vertex_save.shape[0], 1))), axis=1),
                    self.faces_npy,
                    self.uv[0, ...].cpu().detach().numpy(),
                    texture_image=texture_img,  # (H, W, C)
                )
                render_from_path(
                    os.path.join(self.args.OUTPUT_DIR, f"final_mesh.obj"),
                    os.path.join(self.args.OUTPUT_DIR, f"material_0.png"),
                )
                # pickle everything i need to reproduce these visualizations later, int OUTPUT DIR

        if make_videos:
            os.system(
                f"ffmpeg -framerate 10  -i {self.intermediate_folder}/%04d_mesh.png -y {self.args.OUTPUT_DIR}/optim_mesh.mp4"
            )
            os.system(
                f"ffmpeg -framerate 10  -i {self.intermediate_folder}/%04d_tile.png -y {self.args.OUTPUT_DIR}/optim_tile.mp4"
            )
            os.system(
                f"ffmpeg -framerate 10  -i {self.intermediate_folder}/%04d_texture.png -y {self.args.OUTPUT_DIR}/optim_texture.mp4"
            )
            os.system(
                f"ffmpeg -framerate 10  -i {self.intermediate_folder}/%04d_mesh.png -y {self.args.OUTPUT_DIR}/optim_mesh.gif"
            )
            os.system(
                f"ffmpeg -framerate 10  -i {self.intermediate_folder}/%04d_tile.png -y {self.args.OUTPUT_DIR}/optim_tile.gif"
            )
            os.system(
                f"ffmpeg -framerate 10  -i {self.intermediate_folder}/%04d_texture.png -y {self.args.OUTPUT_DIR}/optim_texture.gif"
            )


if __name__ == "__main__":
    escher = Escher()
    if escher.args.EXPERIMENT == "OUTLINE":
        escher.outline()
    elif escher.args.EXPERIMENT == "IMAGE_LOSS":
        escher.run()
