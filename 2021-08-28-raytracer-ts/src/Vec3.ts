import random from './random'

export default class Vec3 {
    x: number
    y: number
    z: number

    constructor(x: number, y: number, z: number) {
        this.x = x
        this.y = y
        this.z = z
    }

    static random(min?: number, max?: number): Vec3 {
        return new Vec3(random(min, max), random(min, max), random(min, max))
    }

    static randomUnitVector(): Vec3 {
        const a = random(0, 2 * Math.PI)
        const z = random(-1, 1)
        const r = Math.sqrt(1 - z * z)
        return new Vec3(r * Math.cos(a), r * Math.sin(a), z)
    }

    negate() {
        return new Vec3(-this.x, -this.y, -this.z)
    }

    invert() {
        return new Vec3(1 / this.x, 1 / this.y, 1 / this.z)
    }

    add(t: Vec3) {
        return new Vec3(this.x + t.x, this.y + t.y, this.z + t.z)
    }

    subtract(t: Vec3) {
        return this.add(t.negate())
    }

    mul(t: Vec3) {
        return new Vec3(this.x * t.x, this.y * t.y, this.z * t.z)
    }

    divide(t: Vec3) {
        return this.mul(t.invert())
    }

    dot(t: Vec3) {
        return this.x * t.x + this.y * t.y + this.z * t.z
    }

    cross(t: Vec3) {
        return new Vec3(
            this.y * t.z - this.z * t.y,
            this.z * t.x - this.x * t.z,
            this.x * t.y - this.y * t.x
        )
    }

    scale(f: number) {
        return new Vec3(this.x * f, this.y * f, this.z * f)
    }

    lengthSquared() {
        return Math.pow(this.x, 2) + Math.pow(this.y, 2) + Math.pow(this.z, 2)
    }

    length() {
        return Math.sqrt(this.lengthSquared())
    }

    toString() {
        return `${this.x} ${this.y} ${this.z}`
    }

    unit() {
        return this.scale(1 / this.length())
    }

    /**
     * Reflect a vector in relation to a normal
     */
    reflect(normal: Vec3) {
        return this.subtract(normal.scale(this.dot(normal)).scale(2))
    }

    /*
     * Refract a vector according to Snell's law
     */
    refract(normal: Vec3, etaRatio: number) {
        const unitVector = this.unit()
        const cosineTheta = unitVector.negate().dot(normal)

        const parallel = normal
            .scale(cosineTheta)
            .add(unitVector)
            .scale(etaRatio)
        const perpendicular = normal.scale(
            -Math.sqrt(1 - parallel.lengthSquared())
        )

        return parallel.add(perpendicular)
    }
}
