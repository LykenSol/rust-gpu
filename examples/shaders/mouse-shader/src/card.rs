use core::{f32::consts::PI, time};
use glam::{Vec2, Vec4, vec2, vec4};
use rand::seq::SliceRandom;
use shared::{glam::Vec3, *};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::{Rectangle, Shape as _, card};

#[derive(Copy, Clone)]
#[repr(u32)]
pub enum Suit {
    Wands = 0,
    Cups = 1,
    Swords = 2,
    Pentacles = 3,
}

impl Suit {
    pub fn suits() -> [Option<Suit>; 5] {
        [
            None,
            Some(Suit::Wands),
            Some(Suit::Cups),
            Some(Suit::Swords),
            Some(Suit::Pentacles),
        ]
    }
}

#[derive(Copy, Clone)]
pub struct Card {
    pub number: u32,
    pub suit: Option<Suit>,
    pub reversed: bool,
}

impl Card {
    pub fn new(number: u32, suit: Option<Suit>, reversed: bool) -> Self {
        Card {
            number,
            suit,
            reversed,
        }
    }

    pub fn reverse_card(mut self) -> Card {
        self.reversed = !self.reversed;
        self
    }
}

// functions that would do things

fn birth_cards(month: u32, day: u32, year: u32) -> (Card, Card) {
    // formula is MM + DD + YY + YY
    // if said munber is < 9 desconds card becomes fisrt + 9 else if it is > 100 the formula is apllied again to obtain a 2 digit number
    let initial_sum = month + day + year / 100 + year % 100;

    let mut first_card = initial_sum / 10 + initial_sum % 10;
    let second_card = if initial_sum < 10 {
        first_card + 9
    } else if initial_sum > 99 {
        first_card = first_card / 10 + first_card % 10;
        first_card / 10 + first_card % 10
    } else {
        first_card / 10 + first_card % 10
    };

    (
        Card::new(first_card, None, false),
        Card::new(second_card, None, false),
    )
}

pub mod render {

    use crate::*;

    #[derive(Copy, Clone)]
    struct BezierCurve {
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
    }

    impl Shape for BezierCurve {
        fn distance(self, p: Vec2) -> f32 {
            let a = self.p1 - self.p0;
            let b = self.p0 - 2.0 * self.p1 + self.p2;
            let c = a * 2.0;
            let d = self.p0 - p;

            let kk = 1.0 / b.dot(b);
            let kx = kk * a.dot(b);
            let ky = kk * (2.0 * a.dot(a) + d.dot(b)) / 3.0;
            let kz = kk * d.dot(a);

            let res;
            let sgn;

            let p = ky - kx * kx;
            let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
            let mut h = q * q + 4.0 * p * p * p;

            fn cro(a: Vec2, b: Vec2) -> f32 {
                a.x * b.y - a.y * b.x
            }

            if h >= 0.0 {
                h = h.sqrt();

                h = h.copysign(q);

                let x = (h - q) / 2.0;
                let v = 1.0_f32.copysign(x) * x.abs().cbrt();
                let mut t = v - p / v;

                t -= (t * (t * t + 3.0 * p) + q) / (3.0 * t * t + 3.0 * p);

                t = (t - kx).clamp(0.0, 1.0);
                let w = d + (c + b * t) * t;
                res = w.dot(w);

                sgn = cro(c + 2.0 * b * t, w);
            } else {
                let z = (-p).sqrt();

                // https://www.shadertoy.com/view/WltSD7
                fn cos_acos_3(mut x: f32) -> f32 {
                    x = (0.5 + 0.5 * x).sqrt();
                    x * (x * (x * (x * -0.008972 + 0.039071) - 0.107074) + 0.576975) + 0.5
                }

                let m = cos_acos_3(q / (p * z * 2.0));
                let mut n = (1.0 - m * m).sqrt();

                n *= 3.0_f32.sqrt();
                let t = (vec3(m + m, -n - m, n - m) * z - kx).clamp(Vec3::ZERO, Vec3::ONE);

                let qx = d + (c + b * t.x) * t.x;
                let dx = qx.dot(qx);
                let sx = cro(a + b * t.x, qx);
                let qy = d + (c + b * t.y) * t.y;
                let dy = qy.dot(qy);
                let sy = cro(a + b * t.y, qy);
                if dx < dy {
                    res = dx;
                    sgn = sx;
                } else {
                    res = dy;
                    sgn = sy;
                }
            }
            res.sqrt().copysign(sgn)
        }
    }

    #[derive(Copy, Clone)]
    struct Ellipse {
        // cpuld have small_radius, big_radius insted of a Vec2
        radiuses: Vec2,
    }

    impl Shape for Ellipse {
        fn distance(mut self, mut p: Vec2) -> f32 {
            fn msign(x: f32) -> f32 {
                if x < 0.0 { -1.0 } else { 1.0 }
            }
            p = p.abs();
            if p.x > p.y {
                // should be .yx()
                p = vec2(p.y, p.x);
                self.radiuses = vec2(self.radiuses.y, self.radiuses.x);
            }
            let l = self.radiuses.y * self.radiuses.y - self.radiuses.x * self.radiuses.x;

            let m = self.radiuses.x * p.x / l;
            let n = self.radiuses.y * p.y / l;
            let m2 = m * m;
            let n2 = n * n;

            let c = (m2 + n2 - 1.0) / 3.0;
            let c3 = c * c * c;

            let d = c3 + m2 * n2;
            let q = d + m2 * n2;
            let g = m + m * n2;
            let mut co;

            if d < 0.0 {
                let h = (q / c3).acos() / 3.0;
                let s = h.cos() + 2.0;
                let t = h.sin() * 3.0_f32.sqrt();
                let rx = (m2 - c * (s + t)).sqrt();
                let ry = (m2 - c * (s - t)).sqrt();
                co = ry + 1.0_f32.copysign(l) * rx + g.abs() / (rx * ry);
            } else {
                let h = 2.0 * m * n * d.sqrt();
                let s = msign(q + h) * (q + h).abs().cbrt();
                let t = msign(q - h) * (q - h).abs().cbrt();
                let rx = -(s + t) - c * 4.0 + 2.0 * m2;
                let ry = (s - t) * 3.0_f32.sqrt();
                let rm = vec2(rx, ry).length();
                co = ry / (rm - rx).sqrt() + 2.0 * g / rm;
            }
            co = (co - m) / 2.0;
            let si = (1.0 - co * co).max(0.0).sqrt();
            let r = self.radiuses * vec2(co, si);

            (r - p).length() * msign(p.y - r.y)
        }
    }

    #[derive(Copy, Clone)]
    struct HalfEllipseLeft {
        // if this words, it could be a generic "half_shape" or soemthing of that nature
        whole_ellipse: Ellipse,
    }

    impl Shape for HalfEllipseLeft {
        fn distance(self, p: Vec2) -> f32 {
            if p.x <= 0.0 {
                self.whole_ellipse.distance(p)
            } else {
                f32::INFINITY
            }
        }
    }

    #[derive(Copy, Clone)]
    struct HalfEllipseRight {
        // if this words, it could be a generic "half_shape" or soemthing of that nature
        whole_ellipse: Ellipse,
    }

    impl Shape for HalfEllipseRight {
        fn distance(self, p: Vec2) -> f32 {
            if p.x >= 0.0 {
                self.whole_ellipse.distance(p)
            } else {
                f32::INFINITY
            }
        }
    }

    #[derive(Copy, Clone)]
    struct HalfEllipseBottom {
        // if this words, it could be a generic "half_shape" or soemthing of that nature
        whole_ellipse: Ellipse,
    }

    impl Shape for HalfEllipseBottom {
        fn distance(self, p: Vec2) -> f32 {
            // i think i need to use cos(radius)
            if p.y
                >= self
                    .whole_ellipse
                    .radiuses
                    .x
                    .max(self.whole_ellipse.radiuses.y)
                    / 2.0
            {
                self.whole_ellipse.distance(p)
            } else {
                f32::INFINITY
            }
        }
    }

    #[derive(Copy, Clone)]
    struct HalfEllipseTop {
        // if this words, it could be a generic "half_shape" or soemthing of that nature
        whole_ellipse: Ellipse,
    }

    impl Shape for HalfEllipseTop {
        fn distance(self, p: Vec2) -> f32 {
            // i think i need to use cos(radius)
            if p.y
                <= -(self
                    .whole_ellipse
                    .radiuses
                    .x
                    .max(self.whole_ellipse.radiuses.y)
                    / 2.0)
            {
                self.whole_ellipse.distance(p)
            } else {
                f32::INFINITY
            }
        }
    }
    #[derive(Copy, Clone)]
    struct HalfEllipseTopActually {
        // if this words, it could be a generic "half_shape" or soemthing of that nature
        whole_ellipse: Ellipse,
    }

    impl Shape for HalfEllipseTopActually {
        fn distance(self, p: Vec2) -> f32 {
            // i think i need to use cos(radius)
            if p.y <= 0.0 {
                self.whole_ellipse.distance(p)
            } else {
                f32::INFINITY
            }
        }
    }

    #[derive(Copy, Clone)]
    struct HalfCircleRight {
        radius: f32,
    }

    impl Shape for HalfCircleRight {
        fn distance(self, p: Vec2) -> f32 {
            if p.x >= 0.0 {
                p.length() - self.radius
            } else {
                f32::INFINITY
            }
        }
    }
    #[derive(Copy, Clone)]
    struct HalfCircleLeft {
        radius: f32,
    }

    impl Shape for HalfCircleLeft {
        fn distance(self, p: Vec2) -> f32 {
            if p.x <= 0.0 {
                p.length() - self.radius
            } else {
                f32::INFINITY
            }
        }
    }
    #[derive(Copy, Clone)]
    struct HalfCircleTop {
        radius: f32,
    }

    impl Shape for HalfCircleTop {
        fn distance(self, p: Vec2) -> f32 {
            if p.y <= 0.0 {
                p.length() - self.radius
            } else {
                Line(vec2(-self.radius, 0.0), vec2(self.radius, 0.0)).distance(p)
            }
        }
    }
    #[derive(Copy, Clone)]
    struct HalfCircleBottom {
        radius: f32,
    }

    impl Shape for HalfCircleBottom {
        fn distance(self, p: Vec2) -> f32 {
            if p.y >= 0.0 {
                p.length() - self.radius
            } else {
                f32::INFINITY
            }
        }
    }

    #[derive(Copy, Clone)]
    struct CutDisk {
        radius: f32,
        cut: f32,
    }

    impl Shape for CutDisk {
        fn distance(self, mut p: Vec2) -> f32 {
            // for lazy resons i will make self.radius = r and self.cut = h
            let r = self.radius;
            let h = self.cut;
            let w = (r * r - h * h).sqrt();

            p.x = p.x.abs();

            //select corner or segment
            let s = ((h - r) * p.x * p.x + w * w * (h + r - 2.0 * p.y)).max(h * p.x - w * p.y);

            if s < 0.0 {
                //circle
                p.length() - self.radius
            } else if p.x < w {
                //segment line
                self.cut - p.y
            } else {
                //segment corner
                (p - vec2(w, h)).length()
            }
        }
    }

    #[derive(Copy, Clone)]
    struct Trapezoid {
        base1: f32,
        base2: f32,
        height: f32,
    }

    impl Shape for Trapezoid {
        fn distance(self, mut p: Vec2) -> f32 {
            let r1 = self.base1;
            let r2 = self.base2;
            let he = self.height;

            let k1 = vec2(r2, he);
            let k2 = vec2(r2 - r1, 2.0 * he);

            p.x = p.x.abs();
            let ca = vec2(
                (p.x - (if p.y < 0.0 { r1 } else { r2 })).max(0.0),
                p.y.abs() - he,
            );
            let cb = p - k1 + k2 * ((k1 - p).dot(k2) / k2.dot(k2)).clamp(0.0, 1.0);

            let s = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };

            s * ca.dot(ca).min(cb.dot(cb)).sqrt()
        }
    }

    #[derive(Copy, Clone)]
    struct Paralelogram {
        wi: f32,
        he: f32,
        sk: f32,
    }

    impl Shape for Paralelogram {
        fn distance(self, p: Vec2) -> f32 {
            let e = vec2(self.sk, self.he);
            let e2 = self.sk * self.sk + self.he * self.he;

            let da = (p.x * e.y - p.y * e.x).abs() - self.wi * self.he;
            let db = p.y.abs() - e.y;
            if da.max(db) < 0.0
            // interior
            {
                db.max(da / e2.sqrt())
            } else
            // exterior
            {
                let f = (p.y / e.y).clamp(-1.0, 1.0);
                let g = (p.x - e.x * f).clamp(-self.wi, self.wi);
                let h = (((p.x - g) * e.x + p.y * e.y) / e2).clamp(-1.0, 1.0);
                (p - vec2(g + e.x * h, e.y * h)).length()
            }
        }
    }

    pub fn cup(radius: f32) -> impl Shape {
        // not sure if i want solid color or a stroke for this, or maybe make a gradient in color later and leve it as solid here
        // add .stroke(radius / 30.0) leter if wanted/needed for scaling purposes
        // also becuse none of the shapes making this is inside anotehr shape the .stroke could be only done on the union
        // for some reason the height of the trapezoid has to be half of the radius evben tho it shoudl be the same height
        CutDisk {
            radius,
            cut: -radius / 2.0,
        }
        .at(vec2(0.0, -radius / 2.0))
        .stroke(radius / 30.0)
        .union(
            Trapezoid {
                base1: radius / 4.0,
                base2: radius / 2.0,
                height: radius / 2.0,
            }
            .at(vec2(0.0, radius)),
        )
        .stroke(radius / 30.0)
    }

    #[derive(Copy, Clone)]
    struct Triangle {
        base: f32,
        height: f32,
    }

    impl Shape for Triangle {
        fn distance(self, mut p: Vec2) -> f32 {
            p.x = p.x.abs();
            let q = vec2(self.base, self.height);
            let a = p - q * (p.dot(q) / q.dot(q)).clamp(0.0, 1.0);
            let b = p - q * vec2((p.x / q.x).clamp(0.0, 1.0), 1.0);
            let k = 1.0_f32.copysign(q.y);
            let d = a.dot(a).min(b.dot(b));
            let s = (k * (p.x * q.y - p.y * q.x)).max(k * (p.y - q.y));

            d.sqrt() * 1.0_f32.copysign(s)
        }
    }

    pub fn sword(length: f32) -> impl Shape {
        Triangle {
            base: length / 40.0,
            height: length / 10.0,
        }
        .at(vec2(
            0.0,
            -(length * 2.0 + length / 10.0 - length * 2.0 / 3.0),
        ))
        .union(
            Trapezoid {
                base1: length / 40.0,
                base2: length / 15.0,
                height: length,
            }
            .at(vec2(0.0, -length + length * 2.0 / 3.0)),
        )
        .union(
            Rectangle {
                size: vec2(length / 2.0, length / 10.0),
            }
            .at(vec2(0.0, length * 2.0 / 3.0)),
        )
        .union(
            Trapezoid {
                base1: length / 20.0,
                base2: length / 40.0,
                height: length / 2.0,
            }
            .at(vec2(0.0, length * 2.0 / 3.0))
            .union(
                Circle {
                    radius: length / 15.0,
                }
                .at(vec2(0.0, length / 2.0 + length * 2.0 / 3.0)),
            ),
        )
    }

    pub fn wand(length: f32) -> impl Shape {
        Triangle {
            base: length / 40.0,
            height: length / 10.0,
        }
        .at(vec2(0.0, -(length * 1.5 + length / 10.0 - length / 2.0)))
        .union(Trapezoid {
            base1: length / 40.0,
            base2: length / 30.0,
            height: length,
        })
        .union(
            Circle {
                radius: length / 15.0,
            }
            .at(vec2(0.0, length)),
        )
        .union(
            Circle {
                radius: length / 17.5,
            }
            .at(vec2(0.0, length / 5.0 + length / 2.0)),
        )
    }

    pub fn card_grid(lengts: Vec2) -> impl Shape {
        Line(vec2(lengts.x, 0.0), vec2(-lengts.x, 0.0))
            .union(Line(
                vec2(lengts.x, lengts.y / 3.0),
                vec2(-lengts.x, lengts.y / 3.0),
            ))
            .union(Line(
                vec2(lengts.x, -lengts.y / 3.0),
                vec2(-lengts.x, -lengts.y / 3.0),
            ))
            .union(Line(
                vec2(lengts.x, -lengts.y / 6.0),
                vec2(-lengts.x, -lengts.y / 6.0),
            ))
            .union(Line(
                vec2(lengts.x, lengts.y / 6.0),
                vec2(-lengts.x, lengts.y / 6.0),
            ))
            .union(Line(
                vec2(lengts.x, -lengts.y * 3.0 / 6.0),
                vec2(-lengts.x, -lengts.y * 3.0 / 6.0),
            ))
            .union(Line(
                vec2(lengts.x, lengts.y * 3.0 / 6.0),
                vec2(-lengts.x, lengts.y * 3.0 / 6.0),
            ))
            .union(Line(vec2(0.0, -lengts.y), vec2(0.0, lengts.y)))
            .union(Line(
                vec2(lengts.x / 3.0, -lengts.y),
                vec2(lengts.x / 3.0, lengts.y),
            ))
            .union(Line(
                vec2(-lengts.x / 3.0, -lengts.y),
                vec2(-lengts.x / 3.0, lengts.y),
            ))
    }

    /*
    #[derive(Copy, Clone)]
    struct Grid<S: Shape, F: Fn(i32, i32) -> S> {
        cell_size: Vec2,
        get_cell: F,
    }

    impl<S: Shape, F: Fn(i32, i32) -> S> Shape for Grid<S, F> {
        fn distance(self, p: Vec2) -> f32 {
            (self.get_cell)(p / self.cell_size).distance(p)
        }
    }
    */

    pub fn pentagram() -> impl Shape {
        let base = 1.0;
        let height = base * (2.0 * PI / 5.0).tan();
        let tri = Triangle { base, height }
            .at(vec2(0.0, -(height + base * (3.0 * PI / 10.0).tan())))
            .stroke(1.0 / 30.0);
        tri.union(tri.rotate(2.0 * PI / 5.0))
            .union(tri.rotate(4.0 * PI / 5.0))
            .union(tri.rotate(6.0 * PI / 5.0))
            .union(tri.rotate(8.0 * PI / 5.0))
    }

    pub fn pentacle() -> impl Shape {
        let radius = 1.0;
        pentagram().scale(radius / 5.0).union(
            Circle {
                radius: radius / 5.0 * ((2.0 * PI / 5.0).tan() + (3.0 * PI / 10.0).tan()),
            }
            .stroke(radius / 25.0),
        )
    }

    #[derive(Copy, Clone)]
    struct Rhombus {
        diagonals: Vec2,
    }

    impl Shape for Rhombus {
        fn distance(self, mut p: Vec2) -> f32 {
            fn ndot(a: Vec2, b: Vec2) -> f32 {
                a.x * b.x - a.y * b.y
            }

            p = p.abs();

            let h = (ndot(self.diagonals - 2.0 * p, self.diagonals)
                / self.diagonals.dot(self.diagonals))
            .clamp(-1.0, 1.0);
            let d = (p - 0.5 * self.diagonals * vec2(1.0 - h, 1.0 + h)).length();

            d * 1.0_f32.copysign(
                p.x * self.diagonals.y + self.diagonals.x * p.y
                    - self.diagonals.x * self.diagonals.y,
            )
        }
    }

    #[derive(Copy, Clone)]
    struct Heart {
        size: f32,
    }

    impl Shape for Heart {
        fn distance(self, mut p: Vec2) -> f32 {
            p.x = p.x.abs();
            Rectangle {
                size: vec2(self.size, self.size),
            }
            .union(
                Circle {
                    radius: self.size / 2.0,
                }
                .at(vec2(0.0, -self.size / 2.0)),
            )
            .rotate(-PI / 4.0)
            .distance(p)
        }
    }

    pub fn page_hat(radius: f32) -> impl Shape {
        Rectangle {
            size: vec2(radius, radius / 3.0),
        }
        .round(radius / 3.0)
        .union(
            Rectangle {
                size: vec2(radius * 1.8, radius / 3.0),
            }
            .at(vec2(0.0, radius / 3.0)),
        )
        .union(
            Trapezoid {
                base1: radius * 2.0,
                base2: radius * 1.5,
                height: radius / 10.0,
            }
            .at(vec2(0.0, radius * 2.0 / 3.0)),
        )
    }

    pub fn knight_helmet(radius: f32) -> impl Shape {
        Circle { radius }
            .intersect(
                Circle {
                    radius: radius * 4.0,
                }
                .at(vec2(0.0, -(radius * 4.0 + radius / 15.0))),
            )
            .intersect(
                Trapezoid {
                    base1: radius / 10.0,
                    base2: radius / 15.0,
                    height: radius / 2.0 + radius / 10.0,
                }
                .at(vec2(0.0, -(radius / 2.0 + radius / 10.0)))
                .invert(),
            )
            .union(
                Rectangle {
                    size: vec2(radius * 2.0, radius / 2.5),
                }
                .intersect(
                    Circle {
                        radius: radius * 4.0,
                    }
                    .invert()
                    .at(vec2(0.0, -radius * 4.0 + radius / 15.0)),
                ),
            )
            .intersect(
                Rhombus {
                    diagonals: vec2(radius / 5.0, radius / 4.0),
                }
                .invert(),
            )
            .union(
                Triangle {
                    base: radius * 4.0 / 6.0,
                    height: radius / 5.0,
                }
                .at(vec2(0.0, -radius / 2.5))
                .hflip(),
            )
            .union(
                Rectangle {
                    size: vec2(radius / 10.0, radius / 2.0 + radius / 5.0),
                }
                .at(vec2(0.0, radius / 2.0 + radius / 10.0)),
            )
            .union(
                Rhombus {
                    diagonals: vec2(radius / 10.0, radius / 8.0),
                }
                .at(vec2(0.0, radius)),
            )
            .union(
                Rectangle {
                    size: vec2(radius / 3.0, radius * 2.0 / 3.0 + radius / 8.0),
                }
                .at(vec2(-(radius - radius / 6.0), radius / 2.0 + radius / 16.0)),
            )
            .union(
                Rectangle {
                    size: vec2(radius / 3.0, radius * 2.0 / 3.0 + radius / 8.0),
                }
                .at(vec2(radius - radius / 6.0, radius / 2.0 + radius / 16.0)),
            )
            .union(
                Circle { radius }
                    .at(vec2(0.0, -radius / 2.0 + radius / 3.0))
                    .intersect(Circle { radius }.at(vec2(0.0, -radius)).invert())
                    .intersect(
                        Rectangle {
                            size: vec2(radius * 2.0 / 3.0, radius * 2.5),
                        }
                        .invert(),
                    )
                    .at(vec2(0.0, radius)),
            )
            .at(vec2(0.0, -radius / 2.0))
    }

    pub fn queen_crown(radius: f32) -> impl Shape {
        Trapezoid {
            base1: radius,
            base2: radius * 0.8,
            height: radius / 2.0,
        }
        .intersect(
            Rhombus {
                diagonals: vec2(radius / 6.0, radius * 2.0 / 6.0),
            }
            .at(vec2(0.0, radius / 12.0))
            .union(
                Rhombus {
                    diagonals: vec2(radius * 0.8 / 6.0, radius * 1.6 / 6.0),
                }
                .rotate(-PI * 30.0 / 180.0)
                .at(vec2(radius * 1.2 / 2.0, radius / 5.0)),
            )
            .union(
                Rhombus {
                    diagonals: vec2(radius * 0.8 / 6.0, radius * 1.6 / 6.0),
                }
                .rotate(PI * 30.0 / 180.0)
                .at(vec2(-radius * 1.2 / 2.0, radius / 5.0)),
            )
            .union(
                Circle {
                    radius: radius / 2.0,
                }
                .at(vec2(radius / 2.0, -radius / 2.0)),
            )
            .union(
                Circle {
                    radius: radius / 2.0,
                }
                .at(vec2(-radius / 2.0, -radius / 2.0)),
            )
            .invert(),
        )
        .union(
            Circle {
                radius: radius / 10.0,
            }
            .at(vec2(0.0, -radius / 2.0)),
        )
        .union(
            Circle {
                radius: radius / 10.0,
            }
            .at(vec2(-radius, -radius / 2.0)),
        )
        .union(
            Circle {
                radius: radius / 10.0,
            }
            .at(vec2(radius, -radius / 2.0)),
        )
    }

    pub fn king_crown(radius: f32) -> impl Shape {
        Rectangle {
            size: vec2(radius * 1.6, radius / 15.0),
        }
        .at(vec2(0.0, radius / 15.0 + radius / 30.0))
        .union(
            Trapezoid {
                base1: radius * 0.9,
                base2: radius * 0.8,
                height: radius / 15.0,
            }
            .at(vec2(0.0, -(radius / 15.0 + radius / 30.0))),
        )
        .union(
            Heart { size: radius }
                .stroke(radius / 20.0)
                .union(Rectangle {
                    size: vec2(radius / 15.0, radius * (PI / 4.0).cos() * 2.0),
                })
                .intersect(
                    Trapezoid {
                        base1: radius * 0.9,
                        base2: radius * 0.8,
                        height: radius / 5.0 + radius / 10.0,
                    }
                    .at(vec2(0.0, radius / 2.0 + radius / 10.0))
                    .invert(),
                )
                .at(vec2(0.0, -radius / 2.0)),
        )
        .union(
            Heart { size: radius / 2.0 }
                .stroke(radius / 20.0)
                .at(vec2(-radius * 0.8, -radius / 2.0))
                .union(
                    Heart { size: radius / 2.0 }
                        .stroke(radius / 20.0)
                        .at(vec2(radius * 0.8, -radius / 2.0)),
                )
                .intersect(
                    Rectangle {
                        size: vec2(radius * 1.6 + radius / 20.0, radius * 1.5),
                    }
                    .invert()
                    .at(vec2(0.0, -radius / 2.0)),
                ),
        )
        .union(
            Trapezoid {
                base1: radius / 8.0,
                base2: radius / 10.0,
                height: radius / 8.0,
            }
            .intersect(
                Circle {
                    radius: radius / 4.0,
                }
                .invert()
                .at(vec2(0.0, -radius / 4.0)),
            )
            .at(vec2(0.0, -radius * (PI / 4.0).cos() * 2.0)),
        )
        .union(
            Circle {
                radius: radius / 4.0,
            }
            .stroke(radius / 30.0)
            .at(vec2(0.0, -(radius * (PI / 4.0).cos() * 2.0 + radius / 4.0))),
        )
        .at(vec2(0.0, radius * 3.0 / 4.0))
    }

    pub fn four_point_star() -> impl Shape {
        Rhombus {
            diagonals: vec2(30.0, 5.0),
        }
        .union(Rhombus {
            diagonals: vec2(5.0, 30.0),
        })
    }

    pub fn handle() -> impl Shape {
        Rectangle {
            size: vec2(2.5, 140.0),
        }
        .union(Circle { radius: 2.5 }.at(vec2(0.0, 70.0)))
        .union(Circle { radius: 2.5 }.at(vec2(0.0, -70.0)))
        .union(Circle { radius: 2.5 }.at(vec2(0.0, -50.0)))
    }

    pub fn drop() -> impl Shape {
        let d: f32 = 0.6;
        let r = 20.0;
        let curve = BezierCurve {
            p0: vec2(0.0, -(8.0 + r)),
            p1: vec2(
                r * (1.0 - d * d).sqrt() / 2.0 + 1.0,
                (-r * d - (8.0 + r)) / 2.0 - 1.0,
            ),
            p2: vec2(r * (1.0 - d * d).sqrt(), -r * d),
        };
        Circle { radius: r }.union(curve.intersect(curve.vflip()).intersect(BezierCurve {
            p0: vec2(r * (1.0 - d * d).sqrt(), -r * d),
            p1: vec2(0.0, 0.0),
            p2: vec2(-r * (1.0 - d * d).sqrt(), -r * d),
        }))
    }

    pub fn bag() -> impl Shape {
        // cirle.halved + triangle + small circle (the knot) + (-esc shape )

        Circle { radius: 20.0 }
            .at(vec2(0.0, 2.5))
            .union(
                Triangle {
                    base: 31.25,
                    height: 50.0,
                }
                .at(vec2(0.0, -30.0)),
            )
            .scale(2.0)
            .union(Circle { radius: 5.0 }.at(vec2(0.0, -60.0)))
            .union(
                Circle { radius: 20.0 }
                    .intersect(
                        Ellipse {
                            radiuses: vec2(15.0, 20.0),
                        }
                        .union(
                            Triangle {
                                base: 10.0,
                                height: 40.0,
                            }
                            .hflip()
                            .at(vec2(0.0, 40.0)),
                        )
                        .invert()
                        .at(vec2(0.0, -15.0)),
                    )
                    .intersect(
                        Triangle {
                            base: 60.0,
                            height: 80.0,
                        }
                        .hflip()
                        .at(vec2(0.0, 30.0)),
                    )
                    .at(vec2(0.0, -85.0)),
            )
    }

    pub fn infinity_syimbol() -> impl Shape {
        drop()
            .stroke(1.0)
            .rotate(PI / 2.0)
            .at(vec2(28.0, 0.0))
            .vmirror()
    }

    pub fn fire() -> impl Shape {
        // a cut circle with like a triangle in the middle kinda
        Circle { radius: 40.0 }
    }

    pub fn magic() -> impl Shape {
        four_point_star()
            .scale(0.3)
            .union(Line(vec2(7.5, 7.5), vec2(10.0, 10.0)))
            .union(Line(vec2(-7.5, 7.5), vec2(-10.0, 10.0)))
            .union(Line(vec2(7.5, -7.5), vec2(10.0, -10.0)))
            .union(Line(vec2(-7.5, -7.5), vec2(-10.0, -10.0)))
    }

    pub fn priestess_crown_outer() -> impl Shape {
        moon()
            .half_shape_y()
            .rotate(-PI / 2.0)
            .at(vec2(62.5, 0.0))
            .union(
                moon()
                    .half_shape_y()
                    .hflip()
                    .rotate(-PI / 2.0)
                    .at(vec2(-62.5, 0.0)),
            )
            .union(
                Rectangle {
                    size: vec2(80.0, 25.0),
                }
                .at(vec2(0.0, 10.0)),
            )
    }

    pub fn six_point_star() -> impl Shape {
        let third = Rhombus {
            diagonals: vec2(5.0, 30.0),
        };
        third
            .union(third.rotate(PI / 3.0))
            .union(third.rotate(-PI / 3.0))
    }

    pub fn emrpess_crown() -> impl Shape {
        six_point_star()
            .scale(0.2)
            .union(six_point_star().scale(0.1).at(vec2(0.0, -10.0)))
            .union(six_point_star().scale(0.1).at(vec2(0.0, -15.0)))
            .union(six_point_star().scale(0.1).at(vec2(0.0, 10.0)))
            .union(six_point_star().scale(0.1).at(vec2(0.0, 15.0)))
            .union(six_point_star().scale(0.1).at(vec2(10.0, 10.0)))
            .union(six_point_star().scale(0.1).at(vec2(-10.0, 10.0)))
            .union(six_point_star().scale(0.1).at(vec2(10.0, 0.0)))
            .union(six_point_star().scale(0.1).at(vec2(-10.0, 0.0)))
            .union(six_point_star().scale(0.1).at(vec2(10.0, -5.0)))
            .union(six_point_star().scale(0.1).at(vec2(-10.0, -5.0)))
            .union(six_point_star().scale(0.1).at(vec2(12.5, 2.5)))
            .union(six_point_star().scale(0.1).at(vec2(-12.5, 2.5)))
            .union(six_point_star().scale(0.1).at(vec2(15.0, 7.5)))
            .union(six_point_star().scale(0.1).at(vec2(-15.0, 7.5)))
            .union(six_point_star().scale(0.1).at(vec2(15.0, 10.0)))
            .union(six_point_star().scale(0.1).at(vec2(-15.0, 10.0)))
            .union(six_point_star().scale(0.1).at(vec2(15.0, 15.0)))
            .union(six_point_star().scale(0.1).at(vec2(-15.0, 15.0)))
            .union(six_point_star().scale(0.1).at(vec2(20.0, 15.0)))
            .union(six_point_star().scale(0.1).at(vec2(-20.0, 15.0)))
            .union(six_point_star().scale(0.1).at(vec2(25.0, 15.0)))
            .union(six_point_star().scale(0.1).at(vec2(-25.0, 15.0)))
            .union(six_point_star().scale(0.1).at(vec2(25.0, 18.0)))
            .union(six_point_star().scale(0.1).at(vec2(-25.0, 18.0)))
            .union(six_point_star().scale(0.1).at(vec2(30.0, 18.0)))
            .union(six_point_star().scale(0.1).at(vec2(-30.0, 18.0)))
            .union(six_point_star().scale(0.1).at(vec2(35.0, 20.0)))
            .union(six_point_star().scale(0.1).at(vec2(-35.0, 20.0)))
            .union(Circle { radius: 30.0 }.stroke(1.0).at(vec2(-45.0, -15.0)))
            .union(Circle { radius: 30.0 }.stroke(1.0).at(vec2(45.0, -15.0)))
            .union(Circle { radius: 60.0 }.stroke(1.0).at(vec2(0.0, 80.0)))
            .intersect(
                Triangle {
                    base: 65.0,
                    height: 100.0,
                }
                .hflip()
                .at(vec2(0.0, 80.0)),
            )
    }

    pub fn emperor_crown() -> impl Shape {
        // tall trapezoin - 3 triangles on top - 1 rombus in the middel to pretend to have a big gem
        let truangle_cut = Triangle {
            base: 45.0,
            height: 40.0,
        }
        .hflip();
        Trapezoid {
            base1: 45.0,
            base2: 35.0,
            height: 25.0,
        }
        .intersect(
            truangle_cut
                .at(vec2(-25.0, 0.0))
                .union(truangle_cut.at(vec2(25.0, 0.0)))
                .union(four_point_star().scale(0.6))
                .invert(),
        )
    }

    pub fn emperor_stick() -> impl Shape {
        Rectangle {
            size: vec2(2.5, 100.0),
        }
        .union(Circle { radius: 2.5 }.at(vec2(0.0, 50.0)))
        .union(Circle { radius: 10.0 }.stroke(1.0).at(vec2(0.0, -50.0)))
        .union(
            Rectangle {
                size: vec2(30.0, 2.5),
            }
            .at(vec2(0.0, -40.0)),
        )
    }

    pub fn hierophant_hat() -> impl Shape {
        let cut_circle = Circle { radius: 60.0 };
        Ellipse {
            radiuses: vec2(30.0, 60.0),
        }
        .intersect(
            cross()
                .scale(0.3)
                .at(vec2(0.0, -30.0))
                .union(
                    cut_circle
                        .stroke(cut_circle.radius / 10.0)
                        .at(vec2(0.0, 30.0)),
                )
                .union(
                    cut_circle
                        .stroke(cut_circle.radius / 8.0)
                        .at(vec2(0.0, 40.0)),
                )
                .union(cut_circle.at(vec2(0.0, 50.0)))
                .invert(),
        )
    }

    pub fn hierophant_stick() -> impl Shape {
        Rectangle {
            size: vec2(2.5, 100.0),
        }
        .union(
            Rectangle {
                size: vec2(20.0, 2.5),
            }
            .at(vec2(0.0, -40.0)),
        )
        .union(
            Rectangle {
                size: vec2(30.0, 2.5),
            }
            .at(vec2(0.0, -30.0)),
        )
        .union(
            Rectangle {
                size: vec2(40.0, 2.5),
            }
            .at(vec2(0.0, -20.0)),
        )
    }

    pub fn lovers_heart() -> impl Shape {
        // wip also Heart may need to be chaged to a function and not be a shape
        Heart { size: 30.0 }
    }

    pub fn hexagram() -> impl Shape {
        let h = 1.0;
        let triangle = Triangle {
            base: h / f32::sqrt(3.0),
            height: h,
        };
        triangle
            .stroke(h / 40.0)
            .at(vec2(0.0, -h * 2.0 / 3.0))
            .union(
                triangle
                    .stroke(h / 40.0)
                    .hflip()
                    .at(vec2(0.0, h * 2.0 / 3.0)),
            )
    }

    pub fn hexagon() -> impl Shape {
        let side = 1.0;
        let triangle = Triangle {
            base: side / 2.0,
            height: side * f32::sqrt(3.0) / 2.0,
        };
        triangle
            .rotate(PI / 6.0)
            .union(triangle.rotate(PI / 2.0))
            .union(triangle.rotate(PI * 5.0 / 6.0))
            .union(triangle.rotate(PI * 7.0 / 6.0))
            .union(triangle.rotate(PI * 9.0 / 6.0))
            .union(triangle.rotate(PI * 11.0 / 6.0))
    }

    pub fn lion() -> impl Shape {
        let triangle_cut = Triangle {
            base: 60.0,
            height: 20.0,
        };
        let spike = Circle { radius: 30.0 }.half_shape_x().hflip().intersect(
            Rectangle {
                size: vec2(30.0, 30.0),
            }
            .at(vec2(42.5, 0.0)),
        );
        let eye = Circle { radius: 7.0 }
            .union(
                Trapezoid {
                    base1: 6.0,
                    base2: 2.5,
                    height: 7.0,
                }
                .at(vec2(-2.0, 7.0)),
            )
            .union(
                Rectangle {
                    size: vec2(6.0, 12.0),
                }
                .at(vec2(-5.0, -3.0))
                .rotate(PI / 6.0),
            )
            .union(
                Rectangle {
                    size: vec2(8.0, 25.0),
                }
                .at(vec2(-7.5, -12.5))
                .rotate(-PI / 12.0),
            );

        let mane = Rhombus {
            diagonals: vec2(35.0, 25.0),
        }
        .round(10.0)
        .at(vec2(0.0, -95.0))
        .union(
            Rectangle {
                size: vec2(40.0, 80.0),
            }
            .rotate(PI / 4.0)
            .at(vec2(60.0, -55.0))
            .intersect(Circle { radius: 26.0 }.at(vec2(19.0, -44.0)).invert()),
        )
        .union(
            Circle { radius: 60.0 }
                .half_shape_x()
                .rotate(-PI / 4.0)
                .at(vec2(70.0, -70.0)),
        )
        .intersect(
            Triangle {
                base: 30.0,
                height: 30.0,
            }
            .hflip()
            .at(vec2(0.0, -105.0))
            .invert(),
        )
        .intersect(
            Rectangle {
                size: vec2(200.0, 25.0),
            }
            .at(vec2(0.0, -126.0))
            .invert(),
        )
        .intersect(
            triangle_cut
                .hflip()
                .rotate(PI / 4.0)
                .at(vec2(57.5, -120.0))
                .union(
                    triangle_cut
                        .scale(2.0)
                        .hflip()
                        .rotate(-PI / 4.0)
                        .at(vec2(32.0, -108.0)),
                )
                .union(
                    Triangle {
                        base: 30.0,
                        height: 40.0,
                    }
                    .rotate(-PI / 5.0)
                    .at(vec2(90.0, -111.0)),
                )
                .union(triangle_cut.rotate(-PI / 6.0).at(vec2(110.0, -67.5)))
                .invert(),
        )
        .union(
            Rectangle {
                size: vec2(100.0, 100.0),
            }
            .at(vec2(82.0, 30.0))
            .intersect(
                Circle { radius: 105.0 }
                    .at(vec2(-20.0, -72.5))
                    .union(
                        Rectangle {
                            size: vec2(10.0, 50.0),
                        }
                        .round(15.0)
                        .rotate(PI / 25.0)
                        .at(vec2(25.0, 35.0)),
                    )
                    .invert(),
            ),
        )
        .union(
            Rectangle {
                size: vec2(40.0, 40.0),
            }
            .at(vec2(80.0, -20.0)),
        )
        .intersect(Circle { radius: 100.0 }.at(vec2(0.0, -20.0)))
        .union(
            Rectangle {
                size: vec2(120.0, 95.5),
            }
            .at(vec2(0.0, 122.5))
            .union(
                Rectangle {
                    size: vec2(80.0, 40.0),
                }
                .rotate(PI / 6.0)
                .at(vec2(58.0, 85.0)),
            )
            .union(
                Rectangle {
                    size: vec2(30.0, 30.0),
                }
                .at(vec2(70.0, 50.0)),
            ),
        )
        .intersect(
            Triangle {
                base: 60.0,
                height: 35.0,
            }
            .rotate(-PI / 2.0)
            .at(vec2(110.0, 70.0))
            .union(
                Rectangle {
                    size: vec2(130.0, 200.0),
                }
                .rotate(-PI / 4.0)
                .at(vec2(90.0, 150.0)),
            )
            .union(
                Triangle {
                    base: 60.0,
                    height: 35.0,
                }
                .rotate(-PI / 2.0)
                .at(vec2(72.5, 145.0)),
            )
            .union(
                Triangle {
                    base: 16.0,
                    height: 32.0,
                }
                .hflip()
                .at(vec2(0.0, 100.0)),
            )
            .union(spike.scale(4.0).at(vec2(-35.0, 100.0)))
            .union(spike.scale(4.0).at(vec2(-97.5, 46.0)))
            .invert(),
        )
        .union(spike.scale(0.7).rotate(-PI / 2.0).at(vec2(46.0, 35.0)))
        .union(
            triangle_cut
                .rotate(-PI / 4.0)
                .scale(0.3)
                .at(vec2(52.5, 72.5)),
        );
        Circle { radius: 22.0 }
            .intersect(
                Rectangle {
                    size: vec2(40.0, 50.0),
                }
                .at(vec2(0.0, 20.0))
                .union(
                    Triangle {
                        base: 20.0,
                        height: 46.0,
                    }
                    .at(vec2(0.0, -45.0)),
                )
                .invert(),
            )
            .union(eye.rotate(-PI / 4.0).at(vec2(32.0, -20.0)).vmirror())
            .union(
                Triangle {
                    base: 30.0,
                    height: 30.0,
                }
                .hflip()
                .at(vec2(0.0, 50.0))
                .intersect(
                    Ellipse {
                        radiuses: vec2(12.0, 36.0),
                    }
                    .rotate(-PI / 4.0)
                    .vmirror()
                    .at(vec2(0.0, 50.0)),
                )
                .intersect(
                    Circle { radius: 30.0 }
                        .at(vec2(33.0, 54.0))
                        .vmirror()
                        .invert(),
                ),
            )
            .union(
                Triangle {
                    base: 32.0,
                    height: 30.0,
                }
                .at(vec2(0.0, 45.0))
                .intersect(
                    Triangle {
                        base: 33.0,
                        height: 35.0,
                    }
                    .invert()
                    .at(vec2(0.0, 48.0)),
                ),
            )
            .union(mane.vmirror())
        // concsving of s trinsglr thingy
    }

    pub fn moon_phases() -> impl Shape {
        //so like for wheel of forune it could have like 6 moon phases on each of a hexagon's trinagle thingy
        let imaginary_circle_radius = 60.0;
        //basicvally how far from the center the moons are placed so they are equally spread
        //and the possitions should be .at(vec2(radius * cos(whatever angle), radius * sin(whatever angle)))
        //could also use this as a generic size for the whole thing and maybe make it = 1.0 and scale from there
        let moon = moon().rotate(PI / 4.0);
        let circle = Circle { radius: 45.0 };
        moon.union(circle.stroke(circle.radius / 40.0))
            .scale(0.2)
            .at(vec2(
                imaginary_circle_radius * f32::cos(PI / 6.0),
                -imaginary_circle_radius * f32::sin(PI / 6.0),
            ))
            .union(circle.intersect(moon.vflip().invert()).scale(0.2).at(vec2(
                imaginary_circle_radius * f32::cos(PI / 6.0),
                imaginary_circle_radius * f32::sin(PI / 6.0),
            )))
            .vmirror()
            .union(
                circle
                    .scale(0.2)
                    .stroke(circle.radius / 40.0)
                    .at(vec2(0.0, -imaginary_circle_radius)),
            )
            .union(circle.scale(0.2).at(vec2(0.0, imaginary_circle_radius)))
            .union(
                Circle { radius: 76.0 }
                    .intersect(hexagon().scale(100.0).rotate(PI / 6.0).stroke(1.0)),
            )
            .union(Circle { radius: 76.0 }.stroke(1.0))
    }

    pub fn scythe_blade() -> impl Shape {
        Rectangle {
            size: vec2(5.0, 9.0),
        }
        .at(vec2(0.0, -60.0))
        .union(
            Circle { radius: 100.0 }.at(vec2(0.0, 30.0)).intersect(
                Circle { radius: 150.0 }
                    .invert()
                    .at(vec2(0.0, 100.0))
                    .half_shape_y(),
            ),
        )
    }

    pub fn chain() -> impl Shape {
        Ellipse {
            radiuses: vec2(5.0, 10.0),
        }
        .stroke(1.0)
        .at(vec2(0.0, -20.0))
        .union(
            Ellipse {
                radiuses: vec2(5.0, 10.0),
            }
            .stroke(1.0),
        )
        .union(
            Ellipse {
                radiuses: vec2(5.0, 10.0),
            }
            .stroke(1.0)
            .at(vec2(0.0, 20.0)),
        )
    }

    pub fn lamp() -> impl Shape {
        Rectangle {
            size: vec2(30.0, 70.0),
        }
        .stroke(1.0)
        .union(
            Trapezoid {
                base1: 30.0,
                base2: 35.0,
                height: 5.0,
            }
            .rotate(PI / 2.0)
            .at(vec2(-20.0, 0.0))
            .stroke(1.0),
        )
        .union(
            Trapezoid {
                base1: 30.0,
                base2: 35.0,
                height: 5.0,
            }
            .rotate(-PI / 2.0)
            .at(vec2(20.0, 0.0))
            .stroke(1.0),
        )
    }

    pub fn cloud() -> impl Shape {
        Circle { radius: 40.0 }
            .at(vec2(-30.0, 0.0))
            .union(Circle { radius: 30.0 }.union(Circle { radius: 15.0 }.at(vec2(30.0, 0.0))))
            .intersect(
                Rectangle {
                    size: vec2(200.0, 60.0),
                }
                .invert()
                .at(vec2(0.0, 40.0)),
            )
            .at(vec2(12.5, 0.0))
    }

    pub fn moon() -> impl Shape {
        Circle { radius: 40.0 }.intersect(Circle { radius: 35.0 }.invert().at(vec2(-10.0, -10.0)))
    }

    pub fn sun() -> impl Shape {
        four_point_star()
            .scale(2.0)
            .union(four_point_star().scale(2.0).rotate(PI / 4.0))
            .union(four_point_star().scale(1.6).rotate(PI / 8.0))
            .union(four_point_star().scale(1.6).rotate(-PI / 8.0))
            .union(Circle { radius: 35.0 })
            .intersect(Circle { radius: 25.0 }.stroke(10.0).invert())
    }

    pub fn cross() -> impl Shape {
        Rectangle {
            size: vec2(20.0, 160.0),
        }
        .union(
            Rectangle {
                size: vec2(100.0, 20.0),
            }
            .at(vec2(0.0, -50.0)),
        )
    }

    pub fn rounded_window() -> impl Shape {
        let size = 1.0;
        Circle { radius: size }
            .at(vec2(0.0, -1.5))
            .union(Rectangle {
                size: vec2(2.0 * size, 3.0 * size),
            })
    }

    pub fn wing() -> impl Shape {
        let fe = Rectangle {
            size: vec2(30.0, 1.0),
        }
        .round(5.0);
        Paralelogram {
            wi: 15.0,
            he: 22.5,
            sk: 15.0,
        }
        .round(5.0)
        .vflip()
        .union(
            fe.at(vec2(45.0, -22.0))
                .union(fe.at(vec2(30.0, -22.0 / 3.0)))
                .union(fe.at(vec2(20.0, 22.0 / 3.0)))
                .union(fe.at(vec2(10.0, 22.0))),
        )
        .at(vec2(25.0, 0.0))
        .rotate(PI / 7.0)
    }

    pub fn cup_rainbow() -> impl Shape {
        let line = Circle { radius: 40.0 }.stroke(1.0).half_shape_x();
        cup(20.0)
            .at(vec2(40.0, 20.0))
            .vmirror()
            .union(line.scale(1.4))
            .union(line.scale(1.2))
            .union(line)
            .union(line.scale(0.8))
            .union(line.scale(0.6))
    }

    pub fn wavy_line() -> impl Shape {
        let curve = BezierCurve {
            p0: vec2(-5.0, 0.0),
            p1: vec2(0.0, -2.5),
            p2: vec2(5.0, 0.0),
        }
        .stroke(1.0);
        curve
            .union(curve.hflip().at(vec2(10.0, 0.0)))
            .union(curve.at(vec2(20.0, 0.0)))
            .union(curve.hflip().at(vec2(30.0, 0.0)))
            .union(curve.at(vec2(40.0, 0.0)))
            .union(curve.hflip().at(vec2(50.0, 0.0)))
            .union(curve.at(vec2(60.0, 0.0)))
            .union(curve.hflip().at(vec2(70.0, 0.0)))
            .vmirror()
    }

    pub fn water() -> impl Shape {
        let wave = wavy_line();
        wave.union(wave.at(vec2(0.0, 7.5)))
            .union(wave.at(vec2(0.0, -7.5)))
    }

    pub fn tower() -> impl Shape {
        let size = 1.0;
        let sqare = Rectangle {
            size: vec2(size / 9.0, size / 9.0),
        };
        let window = rounded_window().scale(size / 10.0);
        Rectangle {
            size: vec2(size, 4.0 * size),
        }
        .union(
            Trapezoid {
                base1: 0.7 * size,
                base2: 0.5 * size,
                height: size / 10.0,
            }
            .at(vec2(0.0, -2.0 * size)),
        )
        .union(
            Rectangle {
                size: vec2(1.4 * size, size / 9.0),
            }
            .at(vec2(0.0, -2.1 * size)),
        )
        .union(
            sqare
                .at(vec2(size * 1.5 / 9.0, -2.25 * size))
                .union(sqare.at(vec2(size * 4.5 / 9.0, -2.25 * size)))
                .vmirror(),
        )
        .intersect(
            window
                .at(vec2(size * 3.0 / 9.0, 1.3 * size))
                .union(window.at(vec2(size * 3.0 / 9.0, -1.3 * size)))
                .union(window.at(vec2(size * 3.0 / 9.0, 0.0)))
                .vmirror()
                .union(window.at(vec2(0.0, 0.65 * size)))
                .union(window.at(vec2(0.0, -0.65 * size)))
                .invert(),
        )
    }

    pub fn lightning_bolt() -> impl Shape {
        let unit = 1.0;
        let para = Paralelogram {
            wi: unit,
            he: 3.0 * unit,
            sk: 1.5 * unit,
        }
        .hflip();
        para.at(vec2(unit / 10.0, -2.0 * unit))
            .union(para.scale(1.5).at(vec2(-unit / 10.0, 3.5 * unit)))
            .intersect(
                Rectangle {
                    size: vec2(6.0 * unit, 15.0 * unit),
                }
                .rotate(-PI / 4.6)
                .at(vec2(-1.8 * unit, 0.0)),
            )
    }

    pub fn tower_clouds() -> impl Shape {
        let cld = cloud().scale(0.4);
        cld.at(vec2(50.0, -40.0))
            .union(cld.at(vec2(53.0, 0.0)))
            .union(cld.at(vec2(-53.0, -25.0)))
            .union(cld.at(vec2(-48.0, 40.0)))
    }

    pub fn tower_lightning_bolts() -> impl Shape {
        let lgtng = lightning_bolt().scale(2.0);
        lgtng
            .at(vec2(55.0, -22.5))
            .union(lgtng.at(vec2(48.0, 17.5)))
            .union(lgtng.at(vec2(-55.0, -7.5)))
            .union(lgtng.at(vec2(-53.0, 57.5)))
    }

    pub fn scales() -> impl Shape {
        // could be like a half circle ot something intersected with a triangle (or just halfcirlce.union triangle idk)
        // a sword for sure somewhere
        // and like chains that hold the 2 cup things
        // so maybe sword in the middle with a line (rectable) on top to hold the 2 halves each with 1 chain and 1 triangle+circle attatched
        let unit = 1.0;
        Circle { radius: unit }
            .half_shape_x()
            .hflip()
            .union(
                Triangle {
                    base: unit,
                    height: unit * 3.0,
                }
                .stroke(unit / 20.0)
                .at(vec2(0.0, -unit * 3.0)),
            )
            .at(vec2(unit * 2.0, unit))
            .union(
                BezierCurve {
                    p0: vec2(0.0, -unit * 3.5),
                    p1: vec2(unit * 1.5, -unit * 3.25),
                    p2: vec2(unit * 2.0, -unit * 3.0),
                }
                .stroke(unit / 20.0),
            )
            .union(chain().scale(unit / 50.0).at(vec2(unit * 2.0, -unit * 2.5)))
            .vmirror()
    }

    pub fn column() -> impl Shape {
        let unit = 1.0;
        Rectangle {
            size: vec2(unit, 8.0 * unit),
        }
        .union(
            Rectangle {
                size: vec2(unit * 1.25, unit / 10.0),
            }
            .at(vec2(0.0, unit * 4.0 + unit / 20.0)),
        )
        .union(
            Rectangle {
                size: vec2(unit * 1.5, unit / 10.0),
            }
            .at(vec2(0.0, unit * 4.0 + unit * 3.0 / 20.0)),
        )
        .hmirror()
    }

    pub fn trumpet() -> impl Shape {
        let unit = 1.0;
        BezierCurve {
            p0: vec2(0.0, -unit * 2.0),
            p1: vec2(unit / 10.0, unit * 1.8),
            p2: vec2(unit / 2.0, unit * 2.0),
        }
        .vmirror()
        .intersect(Rectangle {
            size: vec2(unit * 2.0, unit * 4.0),
        })
    }

    pub fn gravestone() -> impl Shape {
        let unit = 1.0;
        Circle { radius: unit }.at(vec2(0.0, -unit / 2.0)).union(
            Rectangle {
                size: vec2(unit * 2.0, unit * 2.0),
            }
            .at(vec2(0.0, unit / 2.0)),
        )
    }

    pub fn leaf() -> impl Shape {
        let unit = 1.0;
        Rectangle {
            size: vec2(unit / 20.0, unit),
        }
        .at(vec2(0.0, unit / 2.0))
        .union(
            Rectangle {
                size: vec2(unit * 2.0, unit * 2.0),
            }
            .intersect(Circle { radius: unit * 3.0 }.at(vec2(0.0, -unit * 2.5)))
            .intersect(
                Triangle {
                    base: unit * 1.5,
                    height: unit * 1.5,
                }
                .rotate(-PI / 2.0)
                .at(vec2(unit * 1.6, -unit * 1.7))
                .union(
                    Triangle {
                        base: unit * 1.5,
                        height: unit * 1.5,
                    }
                    .rotate(-PI / 2.0)
                    .at(vec2(unit * 2.0, -unit * 1.2)),
                )
                .invert(),
            )
            .at(vec2(0.0, -unit / 2.0)),
        )
        .vmirror()
    }

    pub fn laurel() -> impl Shape {
        let unit = 1.0;

        BezierCurve {
            p0: vec2(0.0, -unit),
            p1: vec2(unit * 0.6, unit * 0.6),
            p2: vec2(0.0, unit),
        }
        .vmirror()
    }

    pub fn laurel_circle() -> impl Shape {
        let initial_laurel = laurel().scale(0.0).rotate(0.0).at(vec2(0.0, 0.0));
        let mut laurels = UnionArray {
            shapes: [initial_laurel; 8],
        };
        let angles = [0.0, 10.0, 20.0, 30.0, 41.0, 54.0, 67.0, 80.0];

        #[allow(clippy::needless_range_loop)]
        for i in 0..laurels.shapes.len() {
            let angle = angles[i] * PI / 180.0;
            laurels.shapes[i] = laurel()
                .scale(6.0)
                .rotate(-angle)
                .at(vec2(angle.cos() * 1.2 * 40.0, angle.sin() * 2.0 * 40.0));
        }
        laurels.hmirror().vmirror().intersect(
            Ellipse {
                radiuses: vec2(1.2 * 40.0 - 1.0, 2.0 * 40.0 - 1.0),
            }
            .stroke(6.0),
        )
    }

    #[derive(Copy, Clone)]
    pub struct CardNumber {
        pub value: u32,
    }

    impl Shape for CardNumber {
        // should work only for I(that should be Ace) -> X (and special cases for page..=king)
        // should be nothing -> XXI for "major arcana"
        fn distance(self, p: Vec2) -> f32 {
            fn i(dx: f32) -> impl Shape {
                Line(vec2(dx, -5.0), vec2(dx, 5.0))
            }

            fn v(dx: f32) -> impl Shape {
                Line(vec2(dx, 5.0), vec2(dx - 2.5, -5.0))
                    .union(Line(vec2(dx, 5.0), vec2(dx + 2.5, -5.0)))
            }

            fn x(dx: f32) -> impl Shape {
                Line(vec2(dx + 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx + 2.5, 5.0)))
            }

            match self.value {
                0 => f32::INFINITY,
                1 => i(0.0).distance(p),
                2 => i(-2.5).union(i(2.5)).distance(p),
                3 => i(-5.0).union(i(0.0)).union(i(5.0)).distance(p),
                4 => i(-5.0).union(v(2.5)).distance(p),
                5 => v(0.0).distance(p),
                6 => v(-5.0).union(i(3.0)).distance(p),
                7 => v(-5.0).union(i(2.5).union(i(7.5))).distance(p),
                8 => v(-7.5)
                    .union(i(0.0).union(i(5.0)).union(i(10.0)))
                    .distance(p),
                9 => i(-5.0).union(x(2.5)).distance(p),
                10 => x(0.0).distance(p),
                11 => x(-5.0).union(i(3.0)).distance(p),
                12 => x(-5.0).union(i(2.5).union(i(7.5))).distance(p),
                13 => x(-7.5)
                    .union(i(0.0).union(i(5.0)).union(i(10.0)))
                    .distance(p),
                14 => x(-7.5).union(i(0.0).union(v(7.5))).distance(p),
                15 => x(-5.0).union(v(5.0)).distance(p),
                16 => x(-7.5).union(v(2.5).union(i(10.0))).distance(p),
                17 => x(-10.0)
                    .union(v(0.0).union(i(7.5).union(i(12.5))))
                    .distance(p),
                18 => x(-12.5)
                    .union(v(-2.5).union(i(5.0).union(i(10.0)).union(i(15.0))))
                    .distance(p),
                19 => x(-7.5).union(i(0.0).union(x(7.5))).distance(p),
                20 => x(-5.0).union(x(5.0)).distance(p),
                21 => x(-7.5).union(x(2.5)).union(i(10.0)).distance(p),
                _ => Line(vec2(0.0, -100.0), vec2(100.0, 100.0)).distance(p),
            }
        }
    }

    #[derive(Copy, Clone)]
    pub struct Letter {
        pub letter: char,
    }

    impl Shape for Letter {
        fn distance(self, p: Vec2) -> f32 {
            fn small_elongated_halfcricle_right(dx: f32, dy: f32) -> impl Shape {
                // dx = radius + lines to the rest of the latter P B type things
                //Circle { radius: 2.5 }.at(vec2(dx, dy)).stroke(0.1)
                HalfCircleRight { radius: 2.5 }
                    .at(vec2(dx, dy))
                    .stroke(0.1)
                    .union(Line(vec2(dx - 2.5, dy + 2.5), vec2(dx, dy + 2.5)))
                    .union(Line(vec2(dx - 2.5, dy - 2.5), vec2(dx, dy - 2.5)))
            }

            fn err(dx: f32) -> impl Shape {
                Line(vec2(dx, 0.0), vec2(dx + 5.0, 0.0))
                    .union(Line(vec2(dx + 2.5, -2.5), vec2(dx + 2.5, 2.5)))
            }

            fn a(dx: f32) -> impl Shape {
                Line(vec2(dx, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx, -5.0), vec2(dx + 2.5, 5.0)))
                    .union(Line(vec2(dx - 1.5, 2.0), vec2(dx + 1.5, 2.0)))
            }

            fn b(dx: f32) -> impl Shape {
                small_elongated_halfcricle_right(dx, -2.5)
                    .union(small_elongated_halfcricle_right(dx, 2.5))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0)))
            }

            fn c(dx: f32) -> impl Shape {
                let whole_ellipse = Ellipse {
                    radiuses: vec2(2.5, 5.0),
                };
                HalfEllipseBottom { whole_ellipse }
                    .at(vec2(dx, 0.0))
                    .stroke(0.1)
                    .union(
                        HalfEllipseTop { whole_ellipse }
                            .union(HalfEllipseLeft { whole_ellipse })
                            .at(vec2(dx, 0.0))
                            .stroke(0.1),
                    )
            }

            fn d(dx: f32) -> impl Shape {
                // i was thinking eitehr making a function that returns distancec if x and y < > 1/2 and thigs of that nature
                let whole_ellipse = Ellipse {
                    radiuses: vec2(2.5, 5.0),
                };
                HalfEllipseRight { whole_ellipse }
                    .at(vec2(dx, 0.0))
                    .stroke(0.1)
                    .union(Line(vec2(dx, -5.0), vec2(dx - 2.5, -5.0)))
                    .union(Line(vec2(dx, 5.0), vec2(dx - 2.5, 5.0)))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0)))
            }

            fn e(dx: f32) -> impl Shape {
                Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx + 2.5, -5.0)))
                    .union(Line(vec2(dx - 2.5, 0.0), vec2(dx + 2.5, 0.0)))
                    .union(Line(vec2(dx - 2.5, 5.0), vec2(dx + 2.5, 5.0)))
            }

            fn e_lower(dx: f32) -> impl Shape {
                let whole_ellipse = Ellipse {
                    radiuses: vec2(2.5, 3.0),
                };
                HalfEllipseBottom { whole_ellipse }
                    .at(vec2(dx, 2.5))
                    .stroke(0.1)
                    .union(
                        HalfEllipseTopActually { whole_ellipse }
                            .union(HalfEllipseLeft { whole_ellipse })
                            .at(vec2(dx, 2.5))
                            .stroke(0.1),
                    )
                    .union(Line(vec2(dx - 2.5, 2.0), vec2(dx + 2.5, 2.0)))
            }

            fn f(dx: f32) -> impl Shape {
                Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx + 2.5, -5.0)))
                    .union(Line(vec2(dx - 2.5, 0.0), vec2(dx + 2.5, 0.0)))
            }

            fn f_lower(dx: f32) -> impl Shape {
                // 2.5/3.0 = 0.833
                HalfCircleTop { radius: 1.0 }
                    .stroke(0.1)
                    .at(vec2(dx, -5.0))
                    .union(Line(
                        vec2(dx - (2.5 - 1.0), -5.0),
                        vec2(dx - (2.5 - 1.0), 5.0),
                    ))
                    .union(Line(vec2(dx - 2.5, -1.0), vec2(dx + 1.25 - 1.0, -1.0)))
            }

            fn g(dx: f32) -> impl Shape {
                let whole_ellipse = Ellipse {
                    radiuses: vec2(2.5, 5.0),
                };
                HalfEllipseBottom { whole_ellipse }
                    .at(vec2(dx, 0.0))
                    .stroke(0.1)
                    .union(
                        HalfEllipseTop { whole_ellipse }
                            .union(HalfEllipseLeft { whole_ellipse })
                            .at(vec2(dx, 0.0))
                            .stroke(0.1),
                    )
                    .union(Line(vec2(dx + 1.0, 0.0), vec2(dx + 2.0, 0.0)))
                    .union(Line(vec2(dx + 2.0, 0.0), vec2(dx + 2.0, 2.0)))
            }

            fn h(dx: f32) -> impl Shape {
                Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx + 2.5, -5.0), vec2(dx + 2.5, 5.0)))
                    .union(Line(vec2(dx - 2.5, 0.0), vec2(dx + 2.5, 0.0)))
            }

            fn h_lower(dx: f32) -> impl Shape {
                HalfCircleTop { radius: 2.5 }
                    .stroke(0.1)
                    .at(vec2(dx, 2.5))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0)))
                    .union(Line(vec2(dx + 2.5, 2.5), vec2(dx + 2.5, 5.0)))
            }

            fn i(dx: f32) -> impl Shape {
                Line(vec2(dx, -5.0), vec2(dx, 5.0))
            }

            fn j(dx: f32) -> impl Shape {
                HalfCircleBottom { radius: 2.5 }
                    .at(vec2(dx, 2.5))
                    .stroke(0.1)
                    .union(Line(vec2(dx + 2.5, -5.0), vec2(dx + 2.5, 2.5)))
            }

            fn k(dx: f32) -> impl Shape {
                Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx - 2.5, 2.5), vec2(dx + 2.5, -5.0)))
                    .union(Line(vec2(dx, -0.5), vec2(dx + 2.5, 5.0)))
            }

            fn l(dx: f32) -> impl Shape {
                // this is L not I
                Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx - 2.5, 5.0), vec2(dx + 2.5, 5.0)))
            }

            fn m(dx: f32) -> impl Shape {
                Line(vec2(dx - 2.5, -5.0), vec2(dx - 5.0, 5.0))
                    .union(Line(vec2(dx + 2.5, -5.0), vec2(dx + 5.0, 5.0)))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx, 2.5)))
                    .union(Line(vec2(dx, 2.5), vec2(dx + 2.5, -5.0)))
            }

            fn n(dx: f32) -> impl Shape {
                Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx + 2.5, -5.0), vec2(dx + 2.5, 5.0)))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx + 2.5, 5.0)))
            }

            fn o(dx: f32) -> impl Shape {
                // to keep the same size character this O might be too skinny
                Ellipse {
                    radiuses: vec2(2.5, 5.0),
                }
                .stroke(0.1)
                .at(vec2(dx, 0.0))
            }

            fn o_lower(dx: f32) -> impl Shape {
                Ellipse {
                    radiuses: vec2(2.5, 3.0),
                }
                .stroke(0.1)
                .at(vec2(dx, 1.5))
            }

            fn p_upper(dx: f32) -> impl Shape {
                small_elongated_halfcricle_right(dx, -2.5)
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0)))
            }

            fn q(dx: f32) -> impl Shape {
                Ellipse {
                    radiuses: vec2(2.5, 5.0),
                }
                .stroke(0.1)
                .at(vec2(dx, 0.0))
                .union(Line(vec2(dx, 1.25), vec2(dx + 2.5, 5.0)))
            }

            fn r(dx: f32) -> impl Shape {
                small_elongated_halfcricle_right(dx, -2.5)
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 5.0)))
                    .union(Line(vec2(dx, 0.0), vec2(dx + 2.5, 5.0)))
            }

            fn s(dx: f32) -> impl Shape {
                HalfCircleTop { radius: 2.5 }
                    .at(vec2(dx, -2.5))
                    .stroke(0.1)
                    .union(
                        HalfCircleLeft { radius: 2.5 }
                            .at(vec2(dx, -2.5))
                            .stroke(0.1),
                    )
                    .union(
                        HalfCircleRight { radius: 2.5 }
                            .at(vec2(dx, 2.5))
                            .stroke(0.1),
                    )
                    .union(
                        HalfCircleBottom { radius: 2.5 }
                            .at(vec2(dx, 2.5))
                            .stroke(0.1),
                    )
            }

            fn t(dx: f32) -> impl Shape {
                Line(vec2(dx, -5.0), vec2(dx, 5.0))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx + 2.5, -5.0)))
            }

            fn u(dx: f32) -> impl Shape {
                HalfCircleBottom { radius: 2.5 }
                    .at(vec2(dx, 2.5))
                    .stroke(0.1)
                    .union(Line(vec2(dx + 2.5, -5.0), vec2(dx + 2.5, 2.5)))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx - 2.5, 2.5)))
            }

            fn v(dx: f32) -> impl Shape {
                Line(vec2(dx, 5.0), vec2(dx - 2.5, -5.0))
                    .union(Line(vec2(dx, 5.0), vec2(dx + 2.5, -5.0)))
            }

            fn w(dx: f32) -> impl Shape {
                Line(vec2(dx - 5.0, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx + 5.0, -5.0), vec2(dx + 2.5, 5.0)))
                    .union(Line(vec2(dx - 2.5, 5.0), vec2(dx, -2.5)))
                    .union(Line(vec2(dx, -2.5), vec2(dx + 2.5, 5.0)))
            }

            fn x(dx: f32) -> impl Shape {
                Line(vec2(dx + 2.5, -5.0), vec2(dx - 2.5, 5.0))
                    .union(Line(vec2(dx - 2.5, -5.0), vec2(dx + 2.5, 5.0)))
            }

            fn y(dx: f32) -> impl Shape {
                Line(vec2(dx, 2.5), vec2(dx - 2.5, -5.0))
                    .union(Line(vec2(dx, 2.0), vec2(dx + 2.5, -5.0)))
                    .union(Line(vec2(dx, 2.0), vec2(dx, 5.0)))
            }

            fn z(dx: f32) -> impl Shape {
                Line(vec2(dx - 2.5, -5.0), vec2(dx + 2.5, -5.0))
                    .union(Line(vec2(dx + 2.5, -5.0), vec2(dx - 2.5, 5.0)))
                    .union(Line(vec2(dx - 2.5, 5.0), vec2(dx + 2.5, 5.0)))
            }
            // self.index should be used here somwhere? i think
            // index being basically based on n aka the length of the string it comes from?
            // +-(n/2 + 0.5*(1+(-1^n))/2)
            match self.letter {
                // ' ' => move xd with 5.0 and basicall go to the next
                'A' => a(0.0).distance(p),
                'B' => b(0.0).distance(p),
                'C' => c(0.0).distance(p),
                'D' => d(0.0).distance(p),
                'E' => e(0.0).distance(p),
                'e' => e_lower(0.0).distance(p),
                'F' => f(0.0).distance(p),
                'f' => f_lower(0.0).distance(p),
                'G' => g(0.0).distance(p),
                'H' => h(0.0).distance(p),
                'h' => h_lower(0.0).distance(p),
                'I' => i(0.0).distance(p),
                'J' => j(0.0).distance(p),
                'K' => k(0.0).distance(p),
                'L' => l(0.0).distance(p),
                'M' => m(0.0).distance(p),
                'N' => n(0.0).distance(p),
                'O' => o(0.0).distance(p),
                'o' => o_lower(0.0).distance(p),
                'P' => p_upper(0.0).distance(p),
                'Q' => q(0.0).distance(p),
                'R' => r(0.0).distance(p),
                'S' => s(0.0).distance(p),
                'T' => t(0.0).distance(p),
                'U' => u(0.0).distance(p),
                'V' => v(0.0).distance(p),
                'W' => w(0.0).distance(p),
                'X' => x(0.0).distance(p),
                'Y' => y(0.0).distance(p),
                'Z' => z(0.0).distance(p),
                _ => err(0.0).distance(p),
            }
        }
    }
    /*
    fn name_string(s: &str) -> impl Shape {
        // char width is 5.0 hights is 10.0
        let letter_size = vec2(5.0, 10.0);
        let letters = s.chars();
        let grind = Grid {
            cell_size: letter_size,
            // or match instead of array indexing
            get_cell: |x, _| letters[x.clamp(0, N) as usize],
        };
        let grind_dist = grind.at(-(letter_size * vec2(N as f32, 1.0) / 2.0));
        let max_num_of_chars = 11 * 15 / 5; // = 33
        let position = max_num_of_chars - s.len();
        // ok lets say i have the string "The MAGICIAN" s.len() = 12, so i have to fill the middle 12 spots, but if 13 then center is a letter
        // so the center aka (0.0, 0.0) of the card is between -5.5 and +5.5 (*5.0 to get the actual size) soo +-(12/2 + 0.5) aka +-(n/2 + 0.5*(1+(-1^n))/2)
        // so for n(even) there are (n-1)/2 on both left and right
        // basically if odd s.len() middle letter is IN (0.0, 0.0), for even s.len() there is no middle Lmid is on -2.5 and Rmid it at 2.5
        let half_len = (s.len() / 2) as f32 - 0.5 * ((1 + (-1_i32).pow(s.len() as u32)) / 2) as f32;
        let mut dx = -half_len * 5.0;
        // idk if there should be an empty thingy thing
        let mut picture_string;
        Line(vec2(0.0, 0.0), vec2(0.0, 0.0)).union(Line(vec2(0.0, 0.0), vec2(0.0, 0.0)));
        for c in s.chars() {
            picture_string.union(picture_char);
            dx += 5.0;
        }
        picture_string
    }
    */
    impl Painter {
        pub fn fill_card(&mut self, card: card::Card, card_center: Vec2) {
            const ALICE_BLUE: Vec4 = vec4(0.941, 0.973, 1.0, 1.0);
            const ANTIQUE_WHITE: Vec4 = vec4(0.980, 0.922, 0.843, 1.0);
            const AZURE: Vec4 = vec4(0.941, 1.0, 1.0, 1.0);
            const BLACK: Vec4 = vec4(0.0, 0.0, 0.0, 1.0);
            const BROWN: Vec4 = vec4(0.376, 0.023, 0.023, 1.0);
            const CRIMSON: Vec4 = vec4(0.863, 0.078, 0.235, 1.0);
            const CYAN: Vec4 = vec4(0.0, 1.0, 1.0, 1.0);
            const DARK_ORANGE: Vec4 = vec4(1.0, 0.549, 0.0, 1.0);
            const GOLD: Vec4 = vec4(1.0, 0.68, 0.0, 1.0);
            const DARK_GRAY: Vec4 = vec4(0.397, 0.397, 0.397, 1.0);
            const DARK_GREEN: Vec4 = vec4(0.0, 0.392, 0.0, 1.0);
            const DARK_GOLDEN_ROD: Vec4 = vec4(0.722, 0.525, 0.043, 1.0);
            const SLATE_GRAY: Vec4 = vec4(0.439, 0.502, 0.565, 1.0);
            const STEEL_BLUE: Vec4 = vec4(0.275, 0.51, 0.706, 1.0);
            const SADDLE_BROWN: Vec4 = vec4(0.545, 0.271, 0.075, 1.0);
            const DARK_SADDLE_BROWN: Vec4 = vec4(0.086, 0.043, 0.012, 1.0);
            const SILVER: Vec4 = vec4(0.527, 0.527, 0.527, 1.0);
            const DEEP_SKY_BLUE: Vec4 = vec4(0.0, 0.749, 1.0, 1.0);
            const WHITE_SMOKE: Vec4 = vec4(0.913, 0.913, 0.913, 1.0);
            if card.suit.is_none() {
                // if reversed
                self.fill_with_contrast_border(
                    card::render::CardNumber { value: card.number }
                        .at(vec2(0.0, -125.0))
                        .at(card_center),
                    BLACK,
                );
                match card.number {
                    0 => {
                        // handle.agled() + a bag?
                        self.fill_with_black_border(
                            handle().rotate(-PI / 3.0).at(card_center),
                            DARK_SADDLE_BROWN,
                        );
                        self.fill_with_black_border(
                            bag().scale(0.5).at(card_center).at(vec2(50.0, 0.0)),
                            BROWN,
                        );
                    }
                    1 => {
                        self.fill_with_black_border(
                            infinity_syimbol().at(card_center).at(vec2(0.0, -40.0)),
                            SILVER,
                        );
                        self.fill_with_black_border(
                            wand(60.0)
                                .rotate(PI / 3.0)
                                .scale(0.8)
                                .at(card_center)
                                .at(vec2(0.0, 40.0)),
                            DARK_SADDLE_BROWN,
                        );
                        self.fill_with_black_border(
                            magic().at(card_center).at(vec2(-60.0, 0.0)),
                            GOLD,
                        );
                    }
                    2 => {
                        self.fill_with_black_border(
                            priestess_crown_outer().at(card_center),
                            ALICE_BLUE,
                        );
                        self.fill_with_black_border(
                            Circle { radius: 25.0 }.at(card_center),
                            ALICE_BLUE,
                        );
                        self.fill_with_black_border(
                            pentagram().scale(4.5).at(card_center),
                            CRIMSON,
                        );
                    }
                    3 => {
                        self.fill_with_black_border(emrpess_crown().at(card_center), GOLD);
                    }
                    4 => {
                        self.fill_with_black_border(
                            emperor_stick()
                                .rotate(PI / 12.0)
                                .scale(0.7)
                                .at(card_center)
                                .at(vec2(-45.0, 20.0)),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            sun().scale(0.3).at(card_center).at(vec2(45.0, 20.0)),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            emperor_crown().at(card_center).at(vec2(0.0, -50.0)),
                            GOLD,
                        );
                    }
                    5 => {
                        self.fill_with_black_border(
                            hierophant_hat().at(card_center).at(vec2(0.0, -30.0)),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            hierophant_stick()
                                .rotate(-PI / 12.0)
                                .at(card_center)
                                .at(vec2(30.0, 20.0)),
                            SADDLE_BROWN,
                        );
                    }
                    6 => {
                        self.fill_with_black_border(
                            sun().scale(0.4).at(card_center).at(vec2(0.0, -80.0)),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            wing()
                                .vmirror()
                                .scale(0.8)
                                .at(card_center)
                                .at(vec2(0.0, -30.0)),
                            WHITE_SMOKE,
                        );
                        self.fill_with_black_border(
                            lovers_heart()
                                .scale(1.2)
                                .at(card_center)
                                .at(vec2(0.0, 30.0)),
                            CRIMSON,
                        );
                    }
                    7 => {
                        self.fill_with_black_border(hexagon().scale(60.0).at(card_center), CRIMSON);
                        self.fill_with_black_border(
                            hexagram()
                                .scale(70.0)
                                .union(hexagon().scale(20.0).rotate(PI / 6.0).stroke(7.0 / 4.0))
                                .at(card_center),
                            DARK_ORANGE,
                        );
                    }
                    8 => {
                        self.fill_with_black_border(
                            infinity_syimbol().at(card_center).at(vec2(0.0, -80.0)),
                            SILVER,
                        );
                        self.fill_with_black_border(
                            lion().scale(0.5).at(card_center).at(vec2(0.0, 15.0)),
                            CRIMSON,
                        );
                    }
                    9 => {
                        self.fill_with_black_border(
                            four_point_star().scale(0.4).at(card_center),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            chain().scale(0.5).at(card_center).at(vec2(0.0, -50.0)),
                            SLATE_GRAY,
                        );
                        self.fill_with_black_border(lamp().at(card_center), STEEL_BLUE);
                        self.fill_with_black_border(
                            handle()
                                .union(handle().scale(0.3).rotate(-PI / 2.0).at(vec2(20.0, -70.0)))
                                .at(card_center)
                                .at(vec2(-40.0, 5.0)),
                            DARK_SADDLE_BROWN,
                        );
                    }
                    10 => {
                        self.fill_with_black_border(Circle { radius: 20.0 }.at(card_center), BLACK);
                        self.fill_with_black_border(moon_phases().at(card_center), ANTIQUE_WHITE);
                        self.fill_with_black_border(
                            Circle { radius: 30.0 }.stroke(27.5).at(card_center),
                            DARK_ORANGE,
                        );
                    }
                    11 => {
                        self.fill_with_black_border(
                            column()
                                .scale(30.0)
                                .at(vec2(50.0, 0.0))
                                .vmirror()
                                .at(card_center),
                            SLATE_GRAY,
                        );
                        self.fill_with_black_border(
                            scales().scale(20.0).at(card_center),
                            ANTIQUE_WHITE,
                        );
                        self.fill_with_black_border(
                            sword(70.0).hflip().at(card_center).at(vec2(0.0, -20.0)),
                            STEEL_BLUE,
                        );
                    }
                    12 => {
                        self.fill_with_black_border(cross().at(card_center), DARK_SADDLE_BROWN);
                        self.fill_with_black_border(
                            chain().scale(0.5).at(card_center).at(vec2(0.0, -40.0)),
                            SLATE_GRAY,
                        );
                    }
                    13 => {
                        self.fill_with_black_border(
                            handle().rotate(-PI / 6.0).at(card_center),
                            DARK_SADDLE_BROWN,
                        );
                        self.fill_with_black_border(
                            scythe_blade().rotate(-PI / 6.0).at(card_center),
                            STEEL_BLUE,
                        );
                    }
                    14 => {
                        self.fill_with_black_border(
                            cup_rainbow()
                                .scale(0.8)
                                .at(card_center)
                                .at(vec2(0.0, -50.0)),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            wing()
                                .at(vec2(2.5, 0.0))
                                .vmirror()
                                .scale(0.8)
                                .at(card_center)
                                .at(vec2(0.0, 52.5)),
                            WHITE_SMOKE,
                        );
                        self.fill_with_black_border(
                            wing()
                                .vmirror()
                                .scale(0.8)
                                .at(card_center)
                                .at(vec2(0.0, 50.0)),
                            WHITE_SMOKE,
                        );
                        self.fill_with_black_border(
                            water().at(card_center).at(vec2(0.0, 80.0)),
                            DEEP_SKY_BLUE,
                        );
                    }
                    15 => {
                        self.fill_with_black_border(
                            moon()
                                .rotate(-PI / 4.0)
                                .at(vec2(0.0, -35.0))
                                .at(card_center),
                            CRIMSON,
                        );
                        self.fill_with_black_border(
                            pentagram()
                                .scale(10.0)
                                .hflip()
                                .at(vec2(0.0, 35.0))
                                .at(card_center),
                            CRIMSON,
                        );
                    }
                    16 => {
                        self.fill_with_black_border(tower().scale(40.0).at(card_center), DARK_GRAY);
                        self.fill_with_black_border(tower_clouds().at(card_center), WHITE_SMOKE);
                        self.fill_with_black_border(tower_lightning_bolts().at(card_center), CYAN);
                    }
                    17 => {
                        self.fill_with_black_border(
                            four_point_star()
                                .rotate(PI / 8.0)
                                .union(four_point_star().rotate(-PI / 8.0))
                                .at(card_center),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            four_point_star()
                                .scale(0.5)
                                .at(vec2(40.0, 50.0))
                                .union(four_point_star().scale(0.5).at(vec2(-40.0, 50.0)))
                                .union(four_point_star().scale(0.5).at(vec2(40.0, -50.0)))
                                .union(four_point_star().scale(0.5).at(vec2(-40.0, -50.0)))
                                .union(four_point_star().scale(0.5).at(vec2(50.0, 0.0)))
                                .union(four_point_star().scale(0.5).at(vec2(-50.0, 0.0)))
                                .at(card_center),
                            AZURE,
                        );
                    }
                    18 => {
                        self.fill_with_black_border(moon().at(card_center), ANTIQUE_WHITE);
                        self.fill_with_black_border(
                            cloud().scale(0.3).at(card_center).at(vec2(-40.0, -40.0)),
                            SILVER,
                        );
                        self.fill_with_black_border(
                            cloud()
                                .scale(0.5)
                                .vflip()
                                .at(card_center)
                                .at(vec2(30.0, 50.0)),
                            SILVER,
                        );
                    }
                    19 => {
                        self.fill_with_black_border(sun().at(card_center), GOLD);
                    }
                    20 => {
                        self.fill_with_black_border(
                            wing()
                                .scale(0.8)
                                .at(vec2(10.0, -50.0))
                                .vmirror()
                                .at(card_center),
                            WHITE_SMOKE,
                        );
                        self.fill_with_black_border(
                            trumpet().scale(25.0).at(card_center).at(vec2(0.0, -20.0)),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            gravestone()
                                .scale(10.0)
                                .at(card_center)
                                .at(vec2(-55.0, 90.0)),
                            DARK_GRAY,
                        );
                        self.fill_with_black_border(
                            gravestone().scale(7.0).at(card_center).at(vec2(0.0, 60.0)),
                            DARK_GRAY,
                        );
                        self.fill_with_black_border(
                            gravestone()
                                .scale(10.0)
                                .at(card_center)
                                .at(vec2(60.0, 80.0)),
                            DARK_GRAY,
                        );
                    }
                    21 => {
                        self.fill_with_black_border(leaf().scale(10.0).at(card_center), DARK_GREEN);
                        self.fill_with_black_border(
                            laurel_circle().scale(0.9).at(card_center),
                            DARK_GREEN,
                        );
                        self.fill_with_black_border(laurel_circle().at(card_center), DARK_GREEN);
                        self.fill_with_black_border(
                            laurel_circle().scale(1.1).at(card_center),
                            DARK_GREEN,
                        );
                    }
                    _ => self.fill_with_black_border(
                        card::render::CardNumber { value: card.number }
                            .scale(4.5)
                            .stroke(4.5)
                            //.at(vec2(0.0, -125.0))
                            .at(card_center),
                        SILVER,
                    ),
                }
                // huge match card.number {} and fill with whatever shapes of whatever colors for all major arcana cards
            } else if card.number < 10 {
                // if reversed
                self.fill_with_contrast_border(
                    card::render::CardNumber {
                        value: card.number + 1,
                    }
                    .stroke(1.0)
                    .at(vec2(0.0, -125.0))
                    .at(card_center),
                    BLACK,
                );
                // if reversed on all
                match card.suit {
                    Some(card::Suit::Wands) => match card.number {
                        0 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0).rotate(PI / 12.0).at(card_center),
                                SADDLE_BROWN,
                            );
                        }
                        1 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 0.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0).at(card_center).at(vec2(30.0, 0.0)),
                                SADDLE_BROWN,
                            );
                        }
                        2 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0).at(card_center),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0).rotate(PI / 6.0).at(card_center),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0).rotate(-PI / 6.0).at(card_center),
                                SADDLE_BROWN,
                            );
                        }
                        3 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .at(card_center)
                                    .at(vec2(-40.0, -10.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .at(card_center)
                                    .at(vec2(40.0, -10.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .at(card_center)
                                    .at(vec2(-25.0, 10.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .at(card_center)
                                    .at(vec2(25.0, 10.0)),
                                SADDLE_BROWN,
                            );
                        }
                        4 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.1)
                                    .at(card_center)
                                    .at(vec2(0.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.9)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, -50.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.9)
                                    .rotate(PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(30.0, -50.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.9)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-15.0, -40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.9)
                                    .rotate(PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(15.0, -40.0)),
                                SADDLE_BROWN,
                            );
                        }
                        5 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0).at(card_center),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0).at(card_center).at(vec2(25.0, 0.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .at(card_center)
                                    .at(vec2(-25.0, 0.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0).at(card_center).at(vec2(50.0, 0.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .at(card_center)
                                    .at(vec2(-50.0, 0.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.1)
                                    .rotate(-PI / 2.5)
                                    .at(card_center),
                                SADDLE_BROWN,
                            );
                        }
                        6 => {
                            // the first 2 should use vflip()
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .rotate(PI - PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -60.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .rotate(PI + PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -60.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(0.0, 50.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(PI / 6.0)
                                    .at(card_center)
                                    .at(vec2(40.0, 35.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 6.0)
                                    .at(card_center)
                                    .at(vec2(-40.0, 35.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(PI / 12.0)
                                    .at(card_center)
                                    .at(vec2(20.0, 45.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 12.0)
                                    .at(card_center)
                                    .at(vec2(-20.0, 45.0)),
                                SADDLE_BROWN,
                            );
                        }
                        7 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, -50.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-10.0, -50.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, -10.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-10.0, -10.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 30.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-10.0, 30.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 70.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-10.0, 70.0)),
                                SADDLE_BROWN,
                            );
                        }
                        8 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(-40.0, -70.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-70.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-50.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-30.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-10.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .hflip()
                                    .at(card_center)
                                    .at(vec2(10.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(30.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(50.0, 40.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(70.0, 40.0)),
                                SADDLE_BROWN,
                            );
                        }
                        9 => {
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(-PI / 4.0)
                                    .at(card_center),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(PI / 4.0)
                                    .at(card_center),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -15.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -15.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -30.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -30.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -45.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -45.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -60.0)),
                                SADDLE_BROWN,
                            );
                            self.fill_with_black_border(
                                card::render::wand(60.0)
                                    .scale(1.5)
                                    .rotate(PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -60.0)),
                                SADDLE_BROWN,
                            );
                        }
                        _ => {}
                    },
                    Some(card::Suit::Cups) => match card.number {
                        0 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        1 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center).at(vec2(30.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        2 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .at(card_center)
                                    .at(vec2(-25.0, -80.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .at(card_center)
                                    .at(vec2(25.0, -80.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center).at(vec2(0.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        3 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, -30.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(0.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(30.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(60.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        4 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center).at(vec2(20.0, 10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center).at(vec2(60.0, 10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .rotate(-PI / 2.5)
                                    .at(card_center)
                                    .at(vec2(60.0, 80.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .rotate(-PI / 2.5)
                                    .at(card_center)
                                    .at(vec2(20.0, 80.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .rotate(PI / 2.5)
                                    .at(card_center)
                                    .at(vec2(-60.0, 80.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        5 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center).at(vec2(-60.0, 0.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center).at(vec2(60.0, 0.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .at(card_center)
                                    .at(vec2(-45.0, -60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .at(card_center)
                                    .at(vec2(45.0, -60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .at(card_center)
                                    .at(vec2(-45.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0).at(card_center).at(vec2(45.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        6 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(0.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-40.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(40.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-20.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(20.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-60.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(60.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        7 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0).scale(0.7).at(card_center),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-30.0, 0.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(30.0, 0.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(0.0, 50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-30.0, 50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(30.0, 50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-60.0, 50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(60.0, 50.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        8 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(0.0, -50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-30.0, -50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(30.0, -50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-60.0, -50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(60.0, -50.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-60.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(60.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-35.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(35.0, -10.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        9 => {
                            self.fill_with_black_border(
                                card::render::cup(20.0).scale(0.6).at(card_center),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-15.0, 30.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(15.0, 30.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(0.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-30.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(30.0, 60.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-15.0, 90.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(15.0, 90.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-45.0, 90.0)),
                                DARK_GOLDEN_ROD,
                            );
                            self.fill_with_black_border(
                                card::render::cup(20.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(45.0, 90.0)),
                                DARK_GOLDEN_ROD,
                            );
                        }
                        _ => {}
                    },
                    Some(card::Suit::Swords) => match card.number {
                        0 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0).at(card_center),
                                STEEL_BLUE,
                            );
                        }
                        1 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .scale(0.7)
                                    .rotate(PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 20.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .scale(0.7)
                                    .rotate(-PI / 4.0)
                                    .at(card_center)
                                    .at(vec2(30.0, 20.0)),
                                STEEL_BLUE,
                            );
                        }
                        2 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.9)
                                    .rotate(PI / 6.0)
                                    .at(card_center),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.9)
                                    .rotate(-PI / 6.0)
                                    .at(card_center),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0).hflip().scale(0.9).at(card_center),
                                STEEL_BLUE,
                            );
                        }
                        3 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-40.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-40.0, 60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-15.0, 0.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .rotate(PI / 2.0)
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(0.0, -85.0)),
                                STEEL_BLUE,
                            );
                        }
                        4 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .rotate(PI / 2.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(0.0, -45.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .rotate(-PI / 2.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(0.0, -15.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .rotate(PI / 2.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(0.0, 15.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .rotate(-PI / 2.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(0.0, 45.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .scale(1.2)
                                    .at(card_center)
                                    .at(vec2(0.0, 20.0)),
                                STEEL_BLUE,
                            );
                        }
                        5 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-70.0, -40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-50.0, -20.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-30.0, -40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(-10.0, -20.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(10.0, -40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.7)
                                    .at(card_center)
                                    .at(vec2(30.0, -20.0)),
                                STEEL_BLUE,
                            );
                        }
                        6 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-10.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(10.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-30.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(30.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-50.0, -40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(50.0, -40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(PI / 4.0)
                                    .scale(0.6)
                                    .at(card_center)
                                    .at(vec2(-30.0, 60.0)),
                                STEEL_BLUE,
                            );
                        }
                        7 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-30.0, -20.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-45.0, -30.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(-60.0, -40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(30.0, -20.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(45.0, -30.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(60.0, -40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .rotate(PI / 4.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(0.0, 40.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .rotate(-PI / 4.0)
                                    .hflip()
                                    .scale(0.8)
                                    .at(card_center)
                                    .at(vec2(0.0, 40.0)),
                                STEEL_BLUE,
                            );
                        }
                        8 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(0.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(20.0, -20.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(-20.0, -20.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(40.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(-40.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(60.0, -30.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(-60.0, -30.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(55.0, 70.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .scale(0.5)
                                    .at(card_center)
                                    .at(vec2(-55.0, 70.0)),
                                STEEL_BLUE,
                            );
                        }
                        9 => {
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center)
                                    .at(vec2(0.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(-PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center)
                                    .at(vec2(0.0, -60.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center)
                                    .at(vec2(0.0, -30.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(-PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center)
                                    .at(vec2(0.0, -30.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(-PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center)
                                    .at(vec2(0.0, 30.0)),
                                STEEL_BLUE,
                            );
                            self.fill_with_black_border(
                                card::render::sword(60.0)
                                    .hflip()
                                    .rotate(-PI / 4.0)
                                    .scale(0.9)
                                    .at(card_center)
                                    .at(vec2(0.0, 30.0)),
                                STEEL_BLUE,
                            );
                        }
                        _ => {}
                    },
                    Some(card::Suit::Pentacles) => match card.number {
                        0 => {
                            self.fill_with_black_border(
                                card::render::pentacle().scale(30.0).at(card_center),
                                GOLD,
                            );
                        }
                        1 => {
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(30.0, -20.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, -20.0)),
                                GOLD,
                            );
                        }
                        2 => {
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -50.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(20.0, -15.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(-20.0, -15.0)),
                                GOLD,
                            );
                        }
                        3 => {
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -70.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(50.0, -20.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(-50.0, -20.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(18.0)
                                    .at(card_center)
                                    .at(vec2(0.0, 70.0)),
                                GOLD,
                            );
                        }
                        4 => {
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -70.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(25.0, -35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-25.0, -35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(50.0, 0.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-50.0, 0.0)),
                                GOLD,
                            );
                        }
                        5 => {
                            self.fill_with_black_border(
                                card::render::pentacle().scale(15.0).at(card_center),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(50.0, 0.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(-50.0, 0.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(25.0, -35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(-25.0, -35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -70.0)),
                                GOLD,
                            );
                        }
                        6 => {
                            self.fill_with_black_border(
                                card::render::pentacle().scale(15.0).at(card_center),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -80.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(0.0, 80.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(-40.0, -40.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(40.0, 40.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(40.0, -40.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(-40.0, 40.0)),
                                GOLD,
                            );
                        }
                        7 => {
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(-40.0, 0.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(40.0, 0.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -90.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(0.0, 90.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(-40.0, -45.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(40.0, 45.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(40.0, -45.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(15.0)
                                    .at(card_center)
                                    .at(vec2(-40.0, 45.0)),
                                GOLD,
                            );
                        }
                        8 => {
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -70.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, -15.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(30.0, -15.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 15.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(30.0, 15.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 45.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(30.0, 45.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(30.0, 75.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-30.0, 75.0)),
                                GOLD,
                            );
                        }
                        9 => {
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(0.0, 35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(50.0, 35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-50.0, 35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(25.0, 0.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-25.0, 0.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(0.0, -35.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(35.0, -105.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-35.0, -105.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(60.0, -70.0)),
                                GOLD,
                            );
                            self.fill_with_black_border(
                                card::render::pentacle()
                                    .scale(12.0)
                                    .at(card_center)
                                    .at(vec2(-60.0, -70.0)),
                                GOLD,
                            );
                        }
                        _ => {}
                    },
                    _ => {}
                }
            } else {
                // if reversed on all
                match card.number {
                    10 => self.fill_with_black_border(
                        card::render::page_hat(40.0)
                            .at(vec2(0.0, -40.0))
                            .at(card_center),
                        DARK_SADDLE_BROWN,
                    ),
                    11 => self.fill_with_black_border(
                        card::render::knight_helmet(40.0)
                            .at(vec2(0.0, -40.0))
                            .at(card_center),
                        SLATE_GRAY,
                    ),
                    12 => self.fill_with_black_border(
                        card::render::queen_crown(40.0)
                            .at(vec2(0.0, -40.0))
                            .at(card_center),
                        GOLD,
                    ),
                    13 => self.fill_with_black_border(
                        card::render::king_crown(40.0)
                            .at(vec2(0.0, -40.0))
                            .at(card_center),
                        GOLD,
                    ),
                    _ => {}
                }
                // if reversed on all
                match card.suit {
                    Some(card::Suit::Wands) => {
                        self.fill_with_black_border(
                            card::render::wand(40.0)
                                .rotate(PI / 12.0)
                                .at(vec2(-55.0, 40.0))
                                .at(card_center),
                            SADDLE_BROWN,
                        );
                        self.fill_with_black_border(
                            card::render::wand(40.0)
                                .rotate(-PI / 12.0)
                                .at(vec2(55.0, 40.0))
                                .at(card_center),
                            SADDLE_BROWN,
                        );
                    }
                    Some(card::Suit::Cups) => {
                        self.fill_with_black_border(
                            card::render::cup(13.33)
                                .at(vec2(-50.0, 40.0))
                                .at(card_center),
                            DARK_GOLDEN_ROD,
                        );
                        self.fill_with_black_border(
                            card::render::cup(13.33)
                                .at(vec2(50.0, 40.0))
                                .at(card_center),
                            DARK_GOLDEN_ROD,
                        );
                    }
                    Some(card::Suit::Swords) => {
                        self.fill_with_black_border(
                            card::render::sword(40.0)
                                .rotate(PI / 3.0)
                                .at(vec2(0.0, 50.0))
                                .at(card_center),
                            STEEL_BLUE,
                        );
                        self.fill_with_black_border(
                            card::render::sword(40.0)
                                .rotate(-PI / 3.0)
                                .at(vec2(0.0, 50.0))
                                .at(card_center),
                            STEEL_BLUE,
                        );
                    }
                    Some(card::Suit::Pentacles) => {
                        self.fill_with_black_border(
                            card::render::pentacle()
                                .scale(20.0)
                                .at(vec2(-50.0, 40.0))
                                .at(card_center),
                            GOLD,
                        );
                        self.fill_with_black_border(
                            card::render::pentacle()
                                .scale(20.0)
                                .at(vec2(50.0, 40.0))
                                .at(card_center),
                            GOLD,
                        );
                    }
                    _ => {}
                }
            }
        }
    }
}

mod my_rand {
    use rand::rand_core::impls::{fill_bytes_via_next, next_u64_via_u32};
    use rand::{Rng, RngCore};

    pub struct Xoshiro128PlusPlus {
        pub(crate) s: [u32; 4],
    }

    impl RngCore for Xoshiro128PlusPlus {
        #[inline]
        fn next_u32(&mut self) -> u32 {
            let res = self.s[0]
                .wrapping_add(self.s[3])
                .rotate_left(7)
                .wrapping_add(self.s[0]);

            let t = self.s[1] << 9;

            self.s[2] ^= self.s[0];
            self.s[3] ^= self.s[1];
            self.s[1] ^= self.s[2];
            self.s[0] ^= self.s[3];

            self.s[2] ^= t;

            self.s[3] = self.s[3].rotate_left(11);

            res
        }

        #[inline]
        fn next_u64(&mut self) -> u64 {
            next_u64_via_u32(self)
        }

        #[inline]
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            fill_bytes_via_next(self, dst);
        }
    }

    pub fn shuffle<T: Copy, const N: usize>(array: &mut [T; N], rng: &mut impl Rng) {
        for i in 0..N {
            let j = (rng.next_u32() as usize) % (i + 1); // rng.random_range(..=i);
            #[allow(clippy::manual_swap)]
            {
                let tmp = array[i];
                array[i] = array[j];
                array[j] = tmp;
            }
        }
    }
}

pub fn cards_demo(painter: &mut crate::Painter, constants: &ShaderConstants) {
    const BLACK: Vec4 = vec4(0.0, 0.0, 0.0, 1.0);
    const GOLD: Vec4 = vec4(1.0, 0.68, 0.0, 1.0);
    const DARK_SILVER: Vec4 = vec4(0.753, 0.753, 0.753, 1.0);

    let testing_suits = false;
    let get_card = |i| {
        if !testing_suits {
            match i {
                0..22 => Card::new(i, None, false),
                _ => Card::new(
                    (i - 22) % 14,
                    Suit::suits()[1 + ((i - 22) / 14) as usize],
                    false,
                ),
            }
        } else {
            Card::new(i % 14, Suit::suits()[4], false)
        }
    };

    // Make mig array thingy with random card numbers
    let mut deck_card_numbers = [0; 78];
    #[allow(clippy::needless_range_loop)]
    for i in 0..deck_card_numbers.len() {
        deck_card_numbers[i] = i as u8;
    }
    let card_timer = constants.time / (1.0 / 3.0) - (constants.time.cos() + 0.95) * 5.0;
    if true {
        let mut rng = my_rand::Xoshiro128PlusPlus {
            s: [1, 2, 3, (card_timer / 71.0) as u32],
        };
        my_rand::shuffle(&mut deck_card_numbers, &mut rng);
    }

    let first_deck_index = (card_timer % 71.0) as usize;
    //let first_deck_index = 17;
    let card_numbers = [
        deck_card_numbers[first_deck_index],
        deck_card_numbers[first_deck_index + 1],
        deck_card_numbers[first_deck_index + 2],
        deck_card_numbers[first_deck_index + 3],
        deck_card_numbers[first_deck_index + 4],
        deck_card_numbers[first_deck_index + 5],
        deck_card_numbers[first_deck_index + 6],
    ];

    // #[cfg(any())]
    #[allow(clippy::needless_range_loop)]
    for i in 0..card_numbers.len() {
        let card = get_card(card_numbers[i] as u32);
        let i = i as f32 - (card_timer % 1.0);

        let card_center = vec2(
            constants.width as f32 / ((card_numbers.len() - 1) as f32) * (i + 0.5),
            constants.height as f32 / 2.0 + (30.0 * (i - ((card_numbers.len() - 1) as f32) / 2.0)),
        );
        let shape = Rectangle {
            size: vec2(11.0, 19.0) * 15.0 - Vec2::splat(20.0 * 2.0),
        }
        .at(card_center)
        .round(20.0);

        painter.drop_shadow(shape, BLACK, 10.0);

        if shape.distance(painter.frag_coord) > 5.0 {
            continue;
        }

        painter.fill_with_contrast_border(
            shape,
            vec4(0.2 - 0.06 * i, 0.04 + 0.1 * i, 0.4 - 0.04 * i, 1.0),
        );

        painter.fill_card(card, card_center);

        //   PAGE of CUPS
        //  KNIGHT of SWORDS
        // QUEEN of PENTACLES
        //   KING of WANDS
        let (string_chars, string_lengths): &'static _ = &const {
            const S: &[&str] = &[
                "",
                "ACE",
                "PAGE",
                "KNIGHT",
                "QUEEN",
                "KING",
                " of ",
                "WANDS",
                "CUPS",
                "SWORDS",
                "PENTACLES",
                "The FOOL",
                "The MAGICIAN",
                "The HIGH PRIESTESS",
                "The EMPRESS",
                "The EMPEROR",
                "The HIEROPHANT",
                "The LOVERS",
                "The CHARIOT",
                "STRENGTH",
                "The HERMIT",
                "WHEEL of FORTUNE",
                "JUSTICE",
                "The HANGED MAN",
                "DEATH",
                "TEMPERANCE",
                "The DEVIL",
                "The TOWER",
                "The STAR",
                "The MOON",
                "The SUN",
                "JUDGEMENT",
                "The WORLD",
            ];
            let mut lengths = [0; S.len()];
            let mut chars = [[' '; 18]; S.len()];
            let mut i = 0;
            while i < S.len() {
                //let mut thik_letter_count = 0;
                let mut j = 0;
                while j < S[i].len() {
                    chars[i][j] = S[i].as_bytes()[j] as char;
                    //if chars[j] == 'M' || chars[j] == 'W' {thik_letter_count += 1;}
                    j += 1;
                }
                lengths[i] = j;
                i += 1;
            }
            (chars, lengths)
        };

        let segments = if card.suit.is_none() {
            [0, 0, 11 + card.number as usize]
        } else {
            match card.number {
                0 => [1, 6, 7 + card.suit.unwrap() as usize],
                10 => [2, 6, 7 + card.suit.unwrap() as usize],
                11 => [3, 6, 7 + card.suit.unwrap() as usize],
                12 => [4, 6, 7 + card.suit.unwrap() as usize],
                13 => [5, 6, 7 + card.suit.unwrap() as usize],
                _ => [0, 0, 0],
            }
        };

        // number + " of " + suit
        let half_len = (string_lengths[segments[0]]
            + string_lengths[segments[1]]
            + string_lengths[segments[2]]) as f32
            / 2.0;
        let mut dx = (-half_len + 0.5) * 7.5;

        #[allow(clippy::needless_range_loop)]
        for segment_index in 0..3 {
            let segment = segments[segment_index];
            for j in 0..string_lengths[segment] {
                let ch = string_chars[segment][j];
                if ch == ' ' {
                    dx += 7.5;
                    continue;
                }
                if ch == 'M' || ch == 'W' {
                    dx += 2.5;
                }

                // if reversed WIP would wotk better with a surface abstraction thingy
                painter.fill_with_black_border(
                    render::Letter { letter: ch }
                        .at(vec2(dx, 100.0))
                        .rotate(if card.reversed { PI } else { 0.0 })
                        .at(card_center),
                    GOLD,
                );

                if ch == 'M' || ch == 'W' {
                    // this being a cheat to make it look better when M or W are there, but not ok beacuse dx is dependent on the size of the "card"
                    dx += 2.5;
                }
                // should += 7.5 i think?
                dx += 7.5;
            }
        }
        if false {
            painter.fill(
                render::card_grid(shape.shape.shape.size).at(card_center),
                DARK_SILVER * Vec3::ONE.extend(0.5),
            );
        }
    }
}
