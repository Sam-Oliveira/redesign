---
layout: page
title: GPU Ray Tracing
description: Ray tracing coursework for the Computer Graphics module at Imperial College London. Received an honourable mention for "scene composition" on Task 2.
img: assets/img/projects/graphics_featured.png
importance: 7
category: coursework
related_publications: false
---

This coursework was the final piece of work for the [Computer Graphics](https://wp.doc.ic.ac.uk/bkainz/teaching/60005-co317-computer-graphics/) module at Imperial College London. I received a 100% mark, and an honourable mention for Task 2.

For Task 1, I implemented a simple GPU-based ray tracing scene, according to a pre-specified scene.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/graphics_featured.png" title="Task 1 ray tracing scene" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Task 1: a pre-specified ray tracing scene.
</div>

For Task 2, we were asked to implement our own scene and add technical extensions to it. Due to my passion for space, I decided to create a solar-system-like scene (ignoring the proper orbits of the planets). Compared to Task 1, I added soft shadows, fog (dark blue), and spherical texture mapping to make each sphere look like the sun or its respective planet.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/graphics_solar_system.png" title="Task 2 solar system scene" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Task 2: an original solar-system scene with soft shadows, fog, and spherical texture mapping.
</div>
