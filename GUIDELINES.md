## Annotation guidelines (Vehicles)

Points the guidelines must settle (decide once, apply everywhere):

- **Box tightness**: The bounding box must tightly enclose the vehicle's visible structure (excluding shadow on the ground, but including bumpers, side mirrors, and wheels if visible).
- **Vehicles partially out of frame**: If a car is cut off by the edge of the frame, annotate only the visible portion. Set the attribute outside to true only when it completely leaves the scene.
- **Keyframe policy**: Place keyframes on frames where the vehicle changes speed, makes a turn, or experiences sudden motion changes. Re-check interpolated segments at least every 25–30 frames.