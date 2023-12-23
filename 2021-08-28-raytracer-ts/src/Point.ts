import Vec3 from './Vec3'
import random from './random'

export default class Point extends Vec3 {
    /**
     * Find a random point inside a unit sphere (which has a length of one)
     */
    static randomInUnitSphere(): Point {
        while (true) {
            const vec = Vec3.random(-1, 1)
            if (vec.lengthSquared() >= 1) {
                continue
            }
            return vec
        }
    }

    /**
     * Find a random point on a unit sphere that is in the same hemisphere as a normal
     */
    static randomInHemisphere(normal: Vec3): Point {
        const inUnitSphere = Point.randomInUnitSphere()
        if (inUnitSphere.dot(normal) > 0) {
            return inUnitSphere
        } else {
            return inUnitSphere.negate()
        }
    }

    /**
     * Find a random point inside a unit disk
     */
    static randomInUnitDisk() {
        while (true) {
            const vec = new Vec3(random(-1, 1), random(-1, 1), 0)
            if (vec.lengthSquared() >= 1) {
                continue
            }
            return vec
        }
    }
}
