"""Glicko-2 rating updates with iterative volatility solve."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GlickoRating:
    rating: float = 1500.0
    rd: float = 350.0
    sigma: float = 0.06

    elo_scale: float = 1500.0
    glicko_scale: float = 173.7178

    @property
    def mu(self) -> float:
        return (self.rating - self.elo_scale) / self.glicko_scale

    @mu.setter
    def mu(self, value: float) -> None:
        self.rating = value * self.glicko_scale + self.elo_scale

    @property
    def phi(self) -> float:
        return self.rd / self.glicko_scale

    @phi.setter
    def phi(self, value: float) -> None:
        self.rd = value * self.glicko_scale


class Glicko2:
    def __init__(self, tau: float = 0.5) -> None:
        self.tau = tau
        self.rating_dict: dict[str, GlickoRating] = {}

    def add_player(
        self, name: str, rating: float = 1500.0, rd: float = 350.0, sigma: float = 0.06
    ) -> None:
        self.rating_dict[name] = GlickoRating(rating=rating, rd=rd, sigma=sigma)

    @staticmethod
    def get_g_factor(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))

    @staticmethod
    def get_expect_result(mu_p1: float, mu_p2: float, g_p2: float) -> float:
        return 1.0 / (1.0 + math.exp(-g_p2 * (mu_p1 - mu_p2)))

    def update_volatility(
        self, vol: float, phi: float, delta: float, v: float, tol: float = 1e-6
    ) -> float:
        a = math.log(vol * vol)

        def f(x: float) -> float:
            ex = math.exp(x)
            term_a = ex * (delta * delta - phi * phi - v - ex)
            term_b = 2.0 * (phi * phi + v + ex) ** 2
            return term_a / term_b - (x - a) / (self.tau * self.tau)

        if delta * delta > phi * phi + v:
            b = math.log(delta * delta - phi * phi - v)
        else:
            k = 1.0
            while f(a - k * self.tau) < 0.0:
                k += 1.0
            b = a - k * self.tau

        A = a
        B = b
        fA = f(A)
        fB = f(B)

        while abs(B - A) > tol:
            C = A + (A - B) * (fA / (fB - fA))
            fC = f(C)
            if fC * fB < 0.0:
                A, fA = B, fB
            else:
                fA *= 0.5
            B, fB = C, fC

        return math.exp(A / 2.0)

    def update_ratings(self, p1_name: str, p2_name: str, score: float) -> None:
        p1 = self.rating_dict[p1_name]
        p2 = self.rating_dict[p2_name]

        p1_g = self.get_g_factor(p1.phi)
        p2_g = self.get_g_factor(p2.phi)

        p1_e = self.get_expect_result(p1.mu, p2.mu, p2_g)
        p2_e = self.get_expect_result(p2.mu, p1.mu, p1_g)

        p1_v = 1.0 / (p2_g * p2_g * p1_e * (1.0 - p1_e))
        p2_v = 1.0 / (p1_g * p1_g * p2_e * (1.0 - p2_e))

        p1_delta = p1_v * p2_g * (score - p1_e)
        p2_delta = p2_v * p1_g * ((1.0 - score) - p2_e)

        p1_sigma = self.update_volatility(p1.sigma, p1.phi, p1_delta, p1_v)
        p2_sigma = self.update_volatility(p2.sigma, p2.phi, p2_delta, p2_v)

        p1_phi_star = math.sqrt(p1.phi * p1.phi + p1_sigma * p1_sigma)
        p2_phi_star = math.sqrt(p2.phi * p2.phi + p2_sigma * p2_sigma)

        p1_phi = 1.0 / math.sqrt((1.0 / (p1_phi_star * p1_phi_star)) + (1.0 / p1_v))
        p2_phi = 1.0 / math.sqrt((1.0 / (p2_phi_star * p2_phi_star)) + (1.0 / p2_v))

        p1_mu = p1.mu + p1_phi * p1_phi * p2_g * (score - p1_e)
        p2_mu = p2.mu + p2_phi * p2_phi * p1_g * ((1.0 - score) - p2_e)

        p1.mu = p1_mu
        p1.phi = p1_phi
        p1.sigma = p1_sigma

        p2.mu = p2_mu
        p2.phi = p2_phi
        p2.sigma = p2_sigma
