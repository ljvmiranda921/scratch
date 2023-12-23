import Color from '../Color'
import Material from '../Material'
import Ray from '../Ray'
import type HitRecord from '../HitRecord'

/*
 * A clear material with reflection
 */
export default class Dielectric extends Material {
    refractiveIndex: number

    constructor(refractiveIndex: number) {
        super()
        this.refractiveIndex = refractiveIndex
    }

    scatter(ray: Ray, hit: HitRecord) {
        // Always (1, 1, 1) because the surface absorbs nothing
        const attenuation = new Color(1, 1, 1)

        const etaRatio = hit.frontFace
            ? 1 / this.refractiveIndex
            : this.refractiveIndex
        const cosineTheta = Math.min(ray.dir.unit().negate().dot(hit.normal), 1)
        const sineTheta = Math.sqrt(1 - Math.pow(cosineTheta, 2))

        // Check if refraction is possible. If not, use reflection
        if (etaRatio * sineTheta > 1) {
            const reflected = ray.dir.reflect(hit.normal)
            const scattered = new Ray(hit.p, reflected)
            return { attenuation, ray: scattered }
        }

        // Based on the angle and the material, there's still a change
        // that ray will be reflected
        const probabilityOfReflection = schlick(cosineTheta, etaRatio)
        if (Math.random() < probabilityOfReflection) {
            const reflected = ray.dir.reflect(hit.normal)
            const scattered = new Ray(hit.p, reflected)
            return { attenuation, ray: scattered }
        }

        const refracted = ray.dir.refract(hit.normal, etaRatio)
        const scattered = new Ray(hit.p, refracted)
        return { attenuation, ray: scattered }
    }
}

/*
 * Schlick approximation
 */
function schlick(cosine: number, refractiveIndex: number) {
    const r0 = (1 - refractiveIndex) / (1 + refractiveIndex)
    return Math.pow(r0, 2) + (1 - Math.pow(r0, 2)) * Math.pow(1 - cosine, 5)
}
