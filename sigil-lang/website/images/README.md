# Image Assets

## og-image.png (REQUIRED)

The `og-image.svg` source file is provided. Convert to PNG before deployment:

```bash
# Using Inkscape
inkscape og-image.svg -w 1200 -h 630 -o og-image.png

# Using ImageMagick
convert og-image.svg og-image.png

# Using rsvg-convert
rsvg-convert -w 1200 -h 630 og-image.svg > og-image.png
```

**Dimensions:** 1200 x 630 pixels (required for Open Graph)

## PWA Icons (REQUIRED for manifest.json)

Generate from `icon.svg`:

```bash
# 192x192 for manifest
inkscape icon.svg -w 192 -h 192 -o icon-192.png

# 512x512 for manifest
inkscape icon.svg -w 512 -h 512 -o icon-512.png

# Apple Touch Icon (180x180)
inkscape icon.svg -w 180 -h 180 -o apple-touch-icon.png
```

## favicon.png (RECOMMENDED)

Create a 32x32 or 64x64 PNG favicon for browsers that don't support SVG data URIs.
