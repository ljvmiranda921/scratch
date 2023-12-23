import Material from '../Material'
import Point from '../Point'
import Ray from '../Ray'
import type Color from '../Color'
import type HitRecord from '../HitRecord'

/**
 * A metal material using reflection
 */
export default class Metal extends Material {
    albedo: Color
    fuzziness: number

    constructor(albedo: Color, fuzziness: number = 0) {
        super()
        this.albedo = albedo
        this.fuzziness = fuzziness
    }

    scatter(ray: Ray, hit: HitRecord) {
        const reflected = ray.dir.unit().reflect(hit.normal)
        const fuzz = Point.randomInUnitSphere().scale(this.fuzziness)
        const scattered = new Ray(hit.p, reflected.add(fuzz))

        if (scattered.dir.dot(hit.normal) <= 0) {
            return null
        }

        return {
            attenuation: this.albedo,
            ray: scattered,
        }
    }
}
