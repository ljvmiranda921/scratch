import Color from './Color'
import type Hittable from './Hittable'
import Point from './Point'
import Vec3 from './Vec3'

export default class Ray {
    orig: Point
    dir: Vec3

    constructor(origin: Point, direction: Vec3) {
        this.orig = origin
        this.dir = direction
    }

    at(t: number) {
        return this.orig.add(this.dir.scale(t))
    }

    /* Return the color of the background (a simple gradient) */
    color(world: Hittable, depth: number): Color {
        // If we've exceeded the ray bounce limit, no more light is gathered
        if (depth <= 0) {
            return new Color(0, 0, 0)
        }

        // Get the first hit on any object in the world
        // The value of t is set to 0.001 to fix "shadow acne"
        const hit = world.hit(this, 0.001, Infinity)

        if (hit) {
            const scattered = hit.material.scatter(this, hit)

            if (scattered) {
                const c = scattered.ray
                    .color(world, depth - 1)
                    .mul(scattered.attenuation)
                return Color.fromVec3(c)
            }
        }

        const unitDirection = this.dir.unit()
        const t = 0.5 * (unitDirection.y + 1.0)
        const c = new Color(1.0, 1.0, 1.0)
            .scale(1.0 - t)
            .add(new Color(0.5, 0.7, 1.0).scale(t))
        return Color.fromVec3(c)
    }
}
