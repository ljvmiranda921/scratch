import Vec3 from './Vec3'

export default class Color extends Vec3 {
    write(): void {
        const ir = Math.floor(256 * clamp(Math.sqrt(this.x), 0.0, 0.999))
        const ig = Math.floor(256 * clamp(Math.sqrt(this.y), 0.0, 0.999))
        const ib = Math.floor(256 * clamp(Math.sqrt(this.z), 0.0, 0.999))
        console.log(`${ir} ${ig} ${ib}`)
    }

    static average(colors: Color[]): Color {
        const sum = colors.reduce(function (acc, color) {
            return acc.add(color) as Color
        }, new Color(0, 0, 0))

        const average = sum.scale(1 / colors.length)
        return Color.fromVec3(average)
    }

    static fromVec3(vec: Vec3): Color {
        return new Color(vec.x, vec.y, vec.z)
    }
}

function clamp(x: number, min: number, max: number): number {
    if (x < min) {
        return min
    }
    if (x > max) {
        return max
    }
    return x
}
