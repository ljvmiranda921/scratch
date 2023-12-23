export default function random(min: number = 0, max: number = 1): number {
    return Math.random() * (max - min) + min
}
