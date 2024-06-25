"""
Baseline experiment where we show that injectivity is important.
In this baselines, points are optimized only with symmetry constraints, without injectivity garantess
See inset figure in the paper.
"""

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

from omegaconf import OmegaConf

import escher.rendering.renderer_nvdiffrast as renderer
import escher.guidance.sd as sd
import escher.geometry.split_square_boundary as split_square_boundary
from escher.misc.color_conversion import hsv2rgb_torch
from escher.geometry.get_base_mesh import to_global_transform, vertex_augmentation, get_empty_2d_square_mesh
from escher.rendering.render_mesh_from_path import render_from_path
from escher.rendering.render_tiling_core import render_mesh_matplotlib
from escher.geometry.save_mesh import save_mesh
from escher.OTE.core.OTESolver import OTESolver
from escher.misc.misc import get_cosine_schedule_with_warmup, seed_everything

start_time = time.time()

# get path to this file
PATH = Path(__file__).parent.absolute()


class Escher:
    def __init__(self) -> None:
        # args = parser.parse_args()
        cli_conf = OmegaConf.from_cli()
        conf_file = cli_conf.get("CONF_FILE", "../configs/base.yaml")
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

        if args.NO_DEFORMATION:
            DEVICE = "cuda:0"

        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.intermediate_folder, exist_ok=True)

        self.init_renderer()
        self.init_stable_diffusion()
        self.init_backgrounds()
        self.get_embeddings()
        self.init_mesh_and_solver()
        self.init_optimizer()

    def init_renderer(self):
        # ================== Init Renderer ===========================
        self.glctx = dr.RasterizeGLContext()
        self.mv = torch.eye(4, device=self.args.DEVICE)[None, ...]
        self.proj = torch.eye(4, device=self.args.DEVICE)[None, ...]

    def init_stable_diffusion(self):
        # ================== Init Stable Diffusion ===========================
        config = sd.Config(
            guidance_scale=self.args.GUIDANCE_SCALE,
            half_precision_weights=self.args.USE_HALF_PRECISION,
            grad_clip=[0, 2.0, 8.0, 1000] if self.args.CLIP_GRADIENTS_IN_SDS else None,
        )
        pprint(config)
        self.model = sd.StableDiffusion(config)

    def get_embeddings(self):
        if isinstance(self.args.PROMPT, str):
            self.args.PROMPT = [self.args.PROMPT]
        assert len(self.args.PROMPT) < 3, "Max two prompts are supported for now"

        with torch.no_grad():
            negative_embedding = self.model.get_text_embeds(self.args.NEGATIVE_PROMPT)
            prompt_embedding = [self.model.get_text_embeds(prompt) for prompt in self.args.PROMPT]
            if len(self.args.PROMPT) == 1:
                self.text_embeds = torch.cat(
                    prompt_embedding * self.args.IMAGE_BATCH_SIZE + [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                )
            else:
                # The text batch is organized as follows
                # [prompt1, ..., prompt1 |  prompt2, ..., prompt2 |  negative, ..., negative |  negative, ..., negative]
                # the corresponding rendering batch is organized as follows
                # [mesh1, ..., mesh1 |  mesh2, ..., mesh2]
                assert self.args.IMAGE_BATCH_SIZE % 2 == 0, "Batch size must be even"
                half_batch_size = self.args.IMAGE_BATCH_SIZE // 2
                positive_embedding = [prompt_embedding[0]] * half_batch_size + [prompt_embedding[1]] * half_batch_size
                negative_embedding = [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                self.text_embeds = torch.cat(positive_embedding + negative_embedding)
                self.half_batch_size = half_batch_size

            del self.model.text_encoder

    def init_backgrounds(self):
        self.black_bg = torch.zeros(1, 3, self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM).cuda()
        self.white_bg = torch.ones(1, 3, self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM)

    def init_mesh_and_solver(self):
        # ============== generate a 2D mesh of a square =======================
        points, faces_npy, split, left, right, top, bottom = get_empty_2d_square_mesh(self.args.MESH_RESOLUTION)

        # =========== init uv ==================================================
        normalized_points = points
        normalized_points = normalized_points - normalized_points.min()
        uv = normalized_points / normalized_points.max()
        uv = torch.from_numpy(uv).unsqueeze(0).to("cuda:0")
        normalized_points = 2 * normalized_points / normalized_points.max() - 1

        # tri = Delaunay(points)
        faces = torch.from_numpy(faces_npy)

        # bdry indices of mesh
        bdry = igl.boundary_loop(faces_npy)

        # split the bdry into 4 sides (left,right,top,down)
        square_sides = split_square_boundary.split_square_boundary(points, bdry)

        # generate nx2 list of edge pairs (i,j)
        adjacency_list = igl.adjacency_list(faces_npy)
        edge_pairs = []
        for r, i in zip(adjacency_list, range(len(adjacency_list))):
            for j in r:
                if i < j:
                    edge_pairs.append((i, j))
        edge_pairs = np.asarray(edge_pairs)

        # scales = torch.nn.Parameter(torch.Tensor(2,1))
        # parameters of the mapping, weights on each edge (used in Tutte's embedding)

        ### prepare the solver that will receive W and return mapped vertices of the mesh
        # uncomment if you want to pin the bdry vertices to place
        # constraint_data =KKTBuilder.pinned_bdry_constraint_matrix(points,bdry)
        # constraints for toric boundary (left connects to right, top to bottom)
        constraint_data = TorusConstraints(points, square_sides)
        # constraint_data = SquareRotateConstraints(points, square_sides)

        ROTATION_MATRIX = constraint_data.get_horizontal_symmetry_orientation().cuda()

        # the solver itself
        solver = OTESolver(edge_pairs, points, constraint_data)
        W = torch.nn.Parameter(torch.randn((edge_pairs.shape[0], 1)))

        if len(self.args.PROMPT) == 2:
            self.faces_1 = faces[split].cuda().type(torch.int32)
            self.faces_2 = faces[~split].cuda().type(torch.int32)

        self.left = torch.nn.Parameter(torch.from_numpy(left).cuda())
        self.right = right
        self.top = torch.nn.Parameter(torch.from_numpy(top).cuda())
        self.bottom = bottom

        self.points = points
        self.uv = uv
        self.bdry = bdry
        self.faces = faces
        self.faces_npy = faces_npy
        self.constraint_data = constraint_data
        self.ROTATION_MATRIX = ROTATION_MATRIX
        self.solver = solver
        self.W = W
        self.square_sides = square_sides

    def init_optimizer(self):
        # ================== Init Texture ===========================
        texture = None
        self.num_channels = 3

        if self.args.OPTIMISE_TEXTURE_MAP:
            self.color_parameters = torch.nn.Parameter(torch.rand(1, 1024, 1024, self.num_channels).to("cuda:0"))
        else:
            self.color_parameters = torch.nn.Parameter(torch.rand(self.W.shape[0], self.num_channels))

        # global_A = torch.nn.Parameter(torch.eye(2,device='cuda:0'))
        self.global_theta = torch.nn.Parameter(torch.zeros(1, device="cuda:0"))
        self.global_sym_ab = torch.nn.Parameter(torch.Tensor([1, 0]).to("cuda:0"))
        # global_theta2 = torch.nn.Parameter(torch.zeros(1,device='cuda:0'))

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.left, "lr": self.args.LR / 10.0},
                {"params": self.top, "lr": self.args.LR / 10.0},
                {"params": self.global_theta, "lr": self.args.LR / 50.0},
                {"params": self.global_sym_ab, "lr": self.args.LR / 200.0},
                {"params": self.color_parameters, "lr": self.args.LR_COLOR / 10.0},
            ],
            lr=self.args.LR,
        )
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 100, int(self.args.N_STEPS * 1.5))

    def init_loss(self):
        # ================== Init Loss ===========================
        if self.args.USE_TARGET_IMAGE:
            self.target_image = torch.from_numpy(np.asarray(PIL.Image.open("./cat.png"))).cuda() / 255

    def render(self, batch_vertices, colors_, texture_batched):
        if len(self.args.PROMPT) == 1:
            return renderer.render_mesh_nvdiffrast(
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
            img_1, _ = renderer.render_mesh_nvdiffrast(
                vertices=batch_vertices.float()[: self.half_batch_size],
                faces=self.faces_1,
                vertices_color=colors_,
                uv=self.uv,
                mv=self.mv,
                proj=self.proj,
                image_size=(self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM),
                glctx=self.glctx,
                texture=texture_batched[: self.half_batch_size],
            )
            img_2, _ = renderer.render_mesh_nvdiffrast(
                vertices=batch_vertices.float()[self.half_batch_size :],
                faces=self.faces_2,
                vertices_color=colors_,
                uv=self.uv,
                mv=self.mv,
                proj=self.proj,
                image_size=(self.args.RENDERER_IMAGE_DIM, self.args.RENDERER_IMAGE_DIM),
                glctx=self.glctx,
                texture=texture_batched[self.half_batch_size :],
            )
            return torch.cat([img_1, img_2], dim=0)

    def run(self):
        # output pbar to stdout
        pbar = tqdm(total=self.args.N_STEPS, desc="steps", position=0)
        # ================== Main training loop ===========================
        for iter in range(self.args.N_STEPS):
            if self.args.CLAMP_TEXTURE:
                self.color_parameters.data = self.color_parameters.data.clip(-1, 1)

            self.optimizer.zero_grad()
            # weights are positive and smaller than 1
            # w = (
            #     torch.special.expit(self.W) * self.args.W_RANGE + (1 - self.args.W_RANGE) / 2
            # )  # [0,1] ----> [r, 1-r] where r = (1-W_RANGE)/2

            # get mapped vertices w.r.t w
            # mapped, _ = self.solver.solve(w)

            # for i in range(2):
            #     mapped[:,i] *= torch.sigmoid(scales[i])*1.5+0.5

            right = self.left + torch.cuda.FloatTensor([2, 0])
            bottom = self.top + torch.cuda.FloatTensor([0, -2])
            mapped = torch.cat([bottom, self.top, right, self.left])

            mapped = mapped.cuda().float()
            mapped_org = mapped
            global_A = torch.eye(2)
            if self.args.GLOBAL_AFFINE and self.constraint_data.allow_global_symmetry():
                #
                global_A = to_global_transform(self.global_theta, self.global_sym_ab)
                mapped = torch.matmul(mapped, torch.transpose(global_A, 0, 1))

            # colors[:, :] = vertices[:, 1:2]
            # colors = (colors + 3) / 6.0 # Roughly normalize to [0, 1]
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
            texture_batched = texture.repeat(self.args.IMAGE_BATCH_SIZE, 1, 1, 1).contiguous()
            texture_batched[drop] = (
                texture_batched[drop] * 0 + torch.rand(self.args.IMAGE_BATCH_SIZE, 1, 1, 1, device=texture.device)[drop]
            )

            # ================== Randomly augment the vertices ===========================
            batch_vertices = vertex_augmentation(
                mapped,
                ROTATION_MATRIX=self.ROTATION_MATRIX,
                RANDOM_RIGID=self.args.RANDOM_RIGID,
                NO_DEFORMATION=self.args.NO_DEFORMATION,
                IMAGE_BATCH_SIZE=self.args.IMAGE_BATCH_SIZE,
            )

            # ================== Special processing if mesh is split in two parts ===========================

            # ================== Render the mesh ===========================
            rendered_img = self.render(batch_vertices, colors_, texture_batched)

            # ================== Post-process the background to be random ===========================
            if self.args.RANDOM_BACKGROUND:
                mask = rendered_img[:, :, :, self.num_channels :].clone()
                # Replace background by a random color
                bg = torch.rand_like(rendered_img)[:, :1, :1, :].expand(rendered_img.shape).contiguous()
                rendered_img = mask * rendered_img + (1 - mask) * bg
                rendered_img[:, :, :, self.num_channels :] = mask

            # ================== Compute loss ===========================
            if self.args.USE_TARGET_IMAGE:
                loss = torch.sum((rendered_img.squeeze()[:, :, : self.num_channels] - self.target_image) ** 2)
            else:
                loss, t = self.model.train_step(
                    rgb=rendered_img[:, :, :, : self.num_channels].contiguous(), text_embeddings=self.text_embeds
                )
            loss = loss + self.args.W_REGULARIZATION * torch.sum(self.W**2)
            # loss = loss + torch.norm(global_Sym/2,'nuc')
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # ================== Progress bar and logging ===========================
            # update the progress bar with per-iteration information
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
            self.visualizer(iter, mapped_org, texture, global_A, batch_vertices, rendered_img)

        # Save final assets
        self.visualizer(iter, mapped_org, texture, global_A, batch_vertices, rendered_img, save_anyway=True)

    def visualizer(self, iter, mapped_org, texture, global_A, batch_vertices, rendered_img, save_anyway=False):
        # Visualizations
        # if (self.args.TILING_RENDER_FREQ > 0 and iter % self.args.TILING_RENDER_FREQ == 0) or save_anyway:
        # if self.args.RANDOM_TILE_COLOR:
        #     render_tiling_different_colors(
        #         mapped_org.cuda(),
        #         self.uv[0, ...],
        #         self.faces,
        #         texture[0, ...],
        #         self.bdry,
        #         self.constraint_data,
        #         torch.matmul(self.ROTATION_MATRIX, global_A),
        #         [os.path.join(self.intermediate_folder, f"{iter}_tile.png"), os.path.join(self.args.OUTPUT_DIR, f"tile.png")],
        #     )
        # else:
        #     render_tiling(
        #         mapped_org.cuda(),
        #         self.uv[0, ...],
        #         self.faces,
        #         texture[0, ...],
        #         self.bdry,
        #         self.constraint_data,
        #         torch.matmul(self.ROTATION_MATRIX, global_A),
        #         [os.path.join(self.intermediate_folder, f"{iter}_tile.png"), os.path.join(self.args.OUTPUT_DIR, f"tile.png")],
        #     )

        if (iter % self.args.VISUALIZATION_FREQ == 0) or save_anyway:
            with torch.no_grad():
                # ============ Log mesh ============
                if not self.args.NO_DEFORMATION:
                    render_mesh_matplotlib(
                        vertices_npy=batch_vertices[0, ...].cpu().detach().numpy(),
                        faces=self.faces_npy,
                        square_sides=self.square_sides,
                        fnames=[
                            os.path.join(self.intermediate_folder, f"{iter}_mesh.png"),
                            os.path.join(self.args.OUTPUT_DIR, f"mesh.png"),
                        ],
                    )

                # ============ Log render ============
                img = rendered_img[:, :, :, : self.num_channels].permute(0, 3, 1, 2)
                img = img.detach().cpu()

                torchvision.utils.save_image(img, os.path.join(self.intermediate_folder, f"{iter}_render.png"))
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
                        texture_img, os.path.join(self.intermediate_folder, f"{iter}_texture.png")
                    )

                R = torch.matmul(self.ROTATION_MATRIX, global_A).cpu().detach().cpu()
                vertex_save = torch.matmul(mapped_org.detach().cpu(), torch.transpose(R, 0, 1))
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
                with open(os.path.join(self.args.OUTPUT_DIR, f"results_{iter}.pkl"), "wb") as f:
                    pickle.dump(
                        {
                            "V": mapped_org.detach().cpu().numpy(),
                            "T": self.faces.cpu().numpy(),
                            "UV": self.uv[0, ...].detach().cpu().numpy(),
                            "texture": texture_img.cpu().detach().numpy(),
                            "bdry": self.bdry,
                            "constraint_data": self.constraint_data,
                            "R": R.numpy(),
                        },
                        f,
                    )


if __name__ == "__main__":
    escher = Escher()
    escher.run()
