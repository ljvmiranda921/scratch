import HitRecord from './HitRecord'
import Hittable from './Hittable'
import type Ray from './Ray'

export default class Hittables extends Hittable {
    objects: Hittable[] = []

    clear() {
        this.objects = []
    }

    add(object: Hittable): void {
        this.objects.push(object)
    }

    /**
     * Return a hit for the closest item in this list of objects
     */
    hit(r: Ray, tMin: number, tMax: number): HitRecord | null {
        let closestSoFar: HitRecord | null = null
        for (const obj of this.objects) {
            const objectHit = obj.hit(r, tMin, closestSoFar?.t ?? tMax)
            if (objectHit) {
                closestSoFar = objectHit
            }
        }
        return closestSoFar
    }
}
