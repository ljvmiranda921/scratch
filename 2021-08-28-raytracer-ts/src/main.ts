import Camera from './Camera'
import Color from './Color'
import Hittables from './Hittables'
import Point from './Point'
import Vec3 from './Vec3'
import randomScene from './scenes/randomScene'
import random from './random'

// Image
const aspectRatio = 16.0 / 9.0
const imageWidth = 400
const imageHeight = Math.floor(imageWidth / aspectRatio)
const samplesPerPixel = 100
const maxDepth = 50

// World

const world: Hittables = randomScene()

// Camera
const camera = new Camera(
    new Point(13, 2, 3), // lookFrom
    new Point(0, 0, 0), // lookAt
    new Vec3(0, 1, 0), // vup
    20, // fov
    aspectRatio,
    0.1, // aperture
    10 // distToFocus
)

// Render
console.log(`P3\n${imageWidth} ${imageHeight}\n255`)

for (let j = imageHeight - 1; j >= 0; j--) {
    process.stderr.clearLine(0)
    process.stderr.cursorTo(0)
    process.stderr.write(`Scanlines remaining: ${j}`)

    for (let i = 0; i < imageWidth; i++) {
        const colorSamples = []

        for (let s = 0; s < samplesPerPixel; s++) {
            const u = (i + random(-1, 1)) / (imageWidth - 1)
            const v = (j + random(-1, 1)) / (imageHeight - 1)

            const r = camera.getRay(u, v)
            const sampledColor = r.color(world, maxDepth)
            colorSamples.push(sampledColor)
        }

        const pixelColor = Color.average(colorSamples)
        pixelColor.write()
    }
}

process.stderr.write(`\nDone.`)
