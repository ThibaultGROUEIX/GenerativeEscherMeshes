import math
import numpy as np


class SparseSystem:
    def __init__(self, n_vertices):
        self.I = []
        self.J = []
        self.V = []
        self.b = []
        self.n_vertices = n_vertices

    def add(self, I, J, V, b):
        self.I.append(I)
        self.J.append(J)
        self.V.append(V)
        self.b.append(b)

    def aggregate(self):
        # I = self.I.copy()
        # J = self.J.copy()
        # V = self.V.copy()
        # b = self.b.copy()
        m = 0
        newI = []
        for i in self.I:
            newI.append(i.flatten() + m)
            m = m + i.max() + 1
        J = [j.flatten() for j in self.J]
        V = [v.flatten() for v in self.V]
        b = [bb.flatten() for bb in self.b]
        return (
            np.concatenate(newI, axis=0),
            np.concatenate(J, axis=0),
            np.concatenate(V, axis=0),
            np.concatenate(b, axis=0),
        )

    def generate_sum_along_line_constraints(self, p1, p2, normal, val):
        p1_x = p1
        p1_y = p1 + self.n_vertices
        p2_x = p2
        p2_y = p2 + self.n_vertices
        # (x_2+x_1)*n_x + (y_2+y_1)*n_y = 0
        J = np.array([p1_x, p2_x, p1_y, p2_y])
        J = np.expand_dims(J, axis=0)
        I = np.array(range(J.shape[0]))
        I = np.stack([I, I, I, I], axis=1)
        b = np.ones(J.shape[0]) * val

        ones = np.ones(J.shape[0])
        V = np.stack([ones * normal[0], ones * normal[0], ones * normal[1], ones * normal[1]], axis=1)
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_rotation_constraints(self, side1, side2, r1, delta):
        # we want to have R*(vert - delta) = horz - delta
        # R =  [c -s
        #        s c]
        # hence we need to satisfy two types of equations
        # (1) c*vert_x - s*vert_y - horz_x = alpha_x
        # (2) s*vert_x + c*vert_y - horz_y = alpha_y
        # with alpha = R*delta - delta
        c = math.cos(r1)
        s = math.sin(r1)
        alpha = -delta
        alpha[0] += c * delta[0] - s * delta[1]
        alpha[1] += s * delta[0] + c * delta[1]
        left_x = side1
        left_y = side1 + self.n_vertices
        top_x = side2
        top_y = side2 + self.n_vertices
        J_1 = np.stack([left_x, left_y, top_x], axis=1)
        I_1 = np.array(range(J_1.shape[0]))
        b_1 = np.tile(alpha[0], I_1.max() + 1)
        I_1 = np.stack([I_1, I_1, I_1], axis=1)
        ones = np.ones(J_1.shape[0])
        V_1 = np.stack([c * ones, -s * ones, -ones], axis=1)

        J_2 = np.stack([left_x, left_y, top_y], axis=1)
        I_2 = np.array(range(J_2.shape[0]))
        b_2 = np.tile(alpha[1], I_2.max() + 1)
        I_2 = np.stack([I_2, I_2, I_2], axis=1) + I_1.max() + 1
        ones = np.ones(J_2.shape[0])
        V_2 = np.stack([s * ones, c * ones, -ones], axis=1)

        I = np.concatenate((I_1, I_2), axis=0)
        J = np.concatenate((J_1, J_2), axis=0)
        V = np.concatenate((V_1, V_2), axis=0)
        b = np.concatenate((b_1, b_2), axis=0)
        assert b.shape == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_straight_line_constraint(self, side, normal):
        side_x = side
        side_y = side + self.n_vertices
        # (x_2-x_1)*n_x + (y_2-y_1)*n_y = 0
        J = np.stack([side_x[1:], side_x[0:-1], side_y[1:], side_y[0:-1]], axis=1)
        I = np.array(range(J.shape[0]))
        I = np.stack([I, I, I, I], axis=1)
        b = np.zeros(J.shape[0])

        ones = np.ones(J.shape[0])
        V = np.stack([ones * normal[0], -ones * normal[0], ones * normal[1], -ones * normal[1]], axis=1)
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_relative_sum_constraint(self, side1, p1, side2, p2, x_axis, sum=0):
        # we want to have reflect*(side2_i - p2) - side1_i + p1 = delta
        # sign is dependent on reflect_x

        sign = 1
        side1_x = side1
        side1_y = side1 + self.n_vertices
        side2_x = side2
        side2_y = side2 + self.n_vertices
        p1_x = p1
        p2_x = p2
        p1_y = p1 + self.n_vertices
        p2_y = p2 + self.n_vertices
        if x_axis:
            J = np.stack([side1_x, np.tile(p1_x, side1_x.shape[0]), side2_x, np.tile(p2_x, side1_x.shape[0])], axis=1)
            I = np.array(range(J.shape[0]))
            b = np.zeros(I.max() + 1) + sum
            I = np.stack([I, I, I, I], axis=1)
            ones = np.ones(J.shape[0])
            V = np.stack([ones * sign, -ones * sign, ones, -ones], axis=1)
        else:
            J = np.stack([side1_y, np.tile(p1_y, side1_y.shape[0]), side2_y, np.tile(p2_y, side1_y.shape[0])], axis=1)
            I = np.array(range(J.shape[0]))
            b = np.zeros(I.max() + 1) + sum
            I = np.stack([I, I, I, I], axis=1)
            ones = np.ones(J.shape[0])
            V = np.stack([ones * sign, -ones * sign, ones, -ones], axis=1)
        assert b.shape[0] == I.max() + 1

        self.add(I, J, V, b)
        return I, J, V, b

    def generate_relative_translation_constraint(self, side1, p1, side2, p2, x_axis, reflect=False, shift=0):
        # we want to have reflect*(side2_i - p2) - side1_i + p1 = delta
        # sign is dependent on reflect_x
        if reflect:
            sign = -1
        else:
            sign = 1
        side1_x = side1
        side1_y = side1 + self.n_vertices
        side2_x = side2
        side2_y = side2 + self.n_vertices
        p1_x = p1
        p2_x = p2
        p1_y = p1 + self.n_vertices
        p2_y = p2 + self.n_vertices
        if x_axis:
            J = np.stack([side1_x, np.tile(p1_x, side1_x.shape[0]), side2_x, np.tile(p2_x, side1_x.shape[0])], axis=1)
            I = np.array(range(J.shape[0]))
            b = np.zeros(I.max() + 1) + shift
            I = np.stack([I, I, I, I], axis=1)
            ones = np.ones(J.shape[0])
            V = np.stack([-ones * sign, ones * sign, ones, -ones], axis=1)
        else:
            J = np.stack([side1_y, np.tile(p1_y, side1_y.shape[0]), side2_y, np.tile(p2_y, side1_y.shape[0])], axis=1)
            I = np.array(range(J.shape[0]))
            b = np.zeros(I.max() + 1) + shift
            I = np.stack([I, I, I, I], axis=1)
            ones = np.ones(J.shape[0])
            V = np.stack([-ones * sign, ones * sign, ones, -ones], axis=1)
        assert b.shape[0] == I.max() + 1

        self.add(I, J, V, b)
        return I, J, V, b

    def generate_translation_constraint_y(self, side1, side2, delta, reflect_x=False):
        # we want to have side2_i - side1_i*sign = delta
        # sign is dependent on reflect_x

        side1_x = side1
        side1_y = side1 + self.n_vertices
        side2_x = side2
        side2_y = side2 + self.n_vertices

        J = np.stack([side1_y, side2_y], axis=1)
        I = np.array(range(J.shape[0]))
        b = np.tile(delta[0], I.max() + 1)
        I = np.stack([I, I], axis=1)
        ones = np.ones(J.shape[0])
        V = np.stack([-ones, ones], axis=1)

        assert b.shape == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_translation_constraint(self, side1, side2, delta, reflect_x=False):
        # we want to have side2_i - side1_i*sign = delta
        # sign is dependent on reflect_x

        side1_x = side1
        side1_y = side1 + self.n_vertices
        side2_x = side2
        side2_y = side2 + self.n_vertices
        J_1 = np.stack([side1_x, side2_x], axis=1)
        I_1 = np.array(range(J_1.shape[0]))
        b_1 = np.tile(delta[0], I_1.max() + 1)
        I_1 = np.stack([I_1, I_1], axis=1)
        ones = np.ones(J_1.shape[0])
        sgn = 1
        if reflect_x:
            sgn = -1
        V_1 = np.stack([-ones, sgn * ones], axis=1)

        J_2 = np.stack([side1_y, side2_y], axis=1)
        I_2 = np.array(range(J_2.shape[0]))
        b_2 = np.tile(delta[1], I_2.max() + 1)
        I_2 = np.stack([I_2, I_2], axis=1) + I_1.max() + 1
        ones = np.ones(J_2.shape[0])
        V_2 = np.stack([-ones, ones], axis=1)

        I = np.concatenate((I_1, I_2), axis=0)
        J = np.concatenate((J_1, J_2), axis=0)
        V = np.concatenate((V_1, V_2), axis=0)
        b = np.concatenate((b_1, b_2), axis=0)
        assert b.shape == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_fixed_constraints_x(self, side, points_x):
        # we want to have side1_i = = points_i
        #

        x = side
        J = x
        I = np.array(range(J.shape[0]))
        b = np.array(points_x)
        ones = np.ones(J.shape[0])
        V = ones
        assert b.shape == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_fixed_constraints_y(self, side, points_y):
        # we want to have side1_i = = points_i
        #
        y = side + self.n_vertices
        J = y
        I = np.array(range(J.shape[0]))
        b = np.array(points_y)
        ones = np.ones(J.shape[0])
        V = ones
        assert b.shape[0] == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_fixed_constraints(self, side, points):
        assert len(points.shape) == 2

        self.generate_fixed_constraints_x(side, points[:, 0])
        self.generate_fixed_constraints_y(side, points[:, 1])
