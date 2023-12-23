import Material from '../Material'
import Point from '../Point'
import Ray from '../Ray'
import type Color from '../Color'
import type HitRecord from '../HitRecord'

/**
 * A diffuse (matte) material
 */
export default class Matte extends Material {
    albedo: Color

    constructor(albedo: Color) {
        super()
        this.albedo = albedo
    }

    scatter(_ray: Ray, hit: HitRecord) {
        const target = hit.p.add(Point.randomInHemisphere(hit.normal))
        const scatterDirection = target.subtract(hit.p)
        const scattered = new Ray(hit.p, scatterDirection)

        return {
            attenuation: this.albedo,
            ray: scattered,
        }
    }
}
