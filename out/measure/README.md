# Measuring the target's pixel size by eye

Nothing here is resized: one pixel on screen is one pixel on the sensor. The
images are single undistorted views from one camera - no warping, no
compositing - so what you see is what the sensor saw.

## Only the middle of the brightness range is shown

The extremes are not measurable and are excluded:

* about 0.4% of every frame sits exactly at the sensor ceiling (963 counts
  after black-level subtraction). A clipped blob's edges are where the
  clipping stopped, not where the object did, so it looks big and shapeless.
* the frame's median is 14 counts, so anything near it is read noise wearing
  a shape.

Everything is selected and stretched on the 90th-99th percentile band
(77-470 counts on burst0500). Blobs that clip, or whose mean level is above
the band, are counted and reported but not shown.

## Files

    <burst>_e01_x<X>y<Y>_f<NNNN>.png   a 1:1 crop, twice: as captured, and with
                                       the fence lattice notched out
    <burst>_e01_x<X>y<Y>_contact.png   the same frames side by side
    <burst>_e01_gallery.png            a SIZE LADDER at 3x: the columns are the
                                       50th, 75th, 90th, 97th and 99.5th
                                       percentile of object size, one row per
                                       frame, with the reference circles at the
                                       same 3x
    context_<burst>_e01.png            the whole frame, for orientation

Objects are also required to be compact (filling at least half their bounding
box, aspect under 2:1), because at this threshold a chain of glints runs
together into one component and measuring its length answers nothing.

## The references

The coloured circles are what a 20 mm object subtends at 300 / 400 / 550 /
700 / 900 mm, given the calibrated f = 1793 px for e01. Match an object
against them and read the depth off the colour.

The yellow bar is the measured mesh period. If you know the mesh pitch, that
bar pins the fence distance without any calibration: distance = pitch * f /
period. At 65 px, a 3.5 mm pitch puts the fence at 96 mm.

## What the two bursts measured

                                   burst0500      burst0300
    objects in the band              1340           1991
    clipped, excluded                 116            226
    median longest side              9 px           8 px
    90th percentile                 27 px          17 px
    99th percentile                 64 px          57 px
    45-90 px and compact          69 (12/frame)  49 (8/frame)

45-90 px is what 20 mm spans between 800 and 400 mm. So the bulk of the
distribution is far too small to be a 20 mm capsule - those are glints - while
the top percentile is the right size, and there are roughly ten of them per
frame.

## Regenerating

    python measure.py burst0300 1 15,40,65 200,150,1100,700
    python gallery.py burst0300 1 90,99          # burst, camera, percentile band

(run from the repository root, with analyze/ on the path)
