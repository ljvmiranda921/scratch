import HitRecord from '../HitRecord'
import Hittable from '../Hittable'
import type Point from '../Point'
import type Ray from '../Ray'
import type Material from '../Material'

export default class Sphere extends Hittable {
    center: Point
    radius: number
    material: Material

    constructor(center: Point, radius: number, material: Material) {
        super()
        this.center = center
        this.radius = radius
        this.material = material
    }

    hit(r: Ray, tMin: number, tMax: number): HitRecord | null {
        const oc = r.orig.subtract(this.center)
        const a = r.dir.lengthSquared()
        const halfB = oc.dot(r.dir)
        const c = oc.lengthSquared() - Math.pow(this.radius, 2)
        const discriminant = Math.pow(halfB, 2) - a * c
        const sqrtD = Math.sqrt(discriminant)

        if (discriminant > 0 && a > 0) {
            let root = (-halfB - sqrtD) / a

            if (root > tMin && root < tMax) {
                const point = r.at(root)
                return new HitRecord(
                    r,
                    root,
                    point.subtract(this.center).scale(1 / this.radius),
                    this.material
                )
            }

            root = (-halfB + sqrtD) / a
            if (root > tMin && root < tMax) {
                const point = r.at(root)
                return new HitRecord(
                    r,
                    root,
                    point.subtract(this.center).scale(1 / this.radius),
                    this.material
                )
            }
        }
        return null
    }
}
