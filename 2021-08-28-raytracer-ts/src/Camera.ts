import Point from './Point'
import Ray from './Ray'
import Vec3 from './Vec3'

export default class Camera {
    origin: Point
    horizontal: Vec3
    vertical: Vec3
    lowerLeftCorner: Vec3
    lensRadius: number

    w: Vec3
    u: Vec3
    v: Vec3

    constructor(
        lookFrom: Point,
        lookAt: Point,
        vup: Vec3,
        vfov: number,
        aspectRatio: number,
        aperture: number,
        focusDistance: number
    ) {
        const theta = toRadians(vfov)
        const h = Math.tan(theta / 2)

        const viewportHeight = 2 * h
        const viewportWidth = aspectRatio * viewportHeight
        const lookDirection = lookFrom.subtract(lookAt)

        this.w = lookDirection.unit()
        this.u = vup.cross(this.w).unit()
        this.v = this.w.cross(this.u)

        this.origin = lookFrom
        this.horizontal = this.u.scale(viewportWidth).scale(focusDistance)
        this.vertical = this.v.scale(viewportHeight).scale(focusDistance)

        this.lowerLeftCorner = this.origin
            .subtract(this.horizontal.scale(0.5))
            .subtract(this.vertical.scale(0.5))
            .subtract(this.w.scale(focusDistance))

        this.lensRadius = aperture / 2
    }

    getRay(s: number, t: number): Ray {
        const rD = Point.randomInUnitDisk().scale(this.lensRadius)
        const offset = this.u.scale(rD.x).add(this.v.scale(rD.y))
        let direction = this.lowerLeftCorner
            .add(this.horizontal.scale(s))
            .add(this.vertical.scale(t))
            .subtract(this.origin)
            .subtract(offset)

        return new Ray(this.origin.add(offset), direction)
    }
}

function toRadians(degrees: number) {
    return degrees * (Math.PI / 180)
}
