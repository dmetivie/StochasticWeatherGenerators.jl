# 🧰 Utilities

## Climate indexes

```@docs
VCX3
monthly_agg
corTail
longuest_spell
pmf_spell
```

## Generic utilities

```@docs
my_color
```

## Map utilities

```@docs
dms_to_dd
haversine
LLA
distance_x_to_y
```

## Plots

The plot functions are defined once a Makie backend is loaded, e.g. `using CairoMakie`.
`map_with_stations` also needs `using GeoMakie, NaturalEarth`.

```@docs
errorline
errorscatter
errorlinehist
errorscatterhist
map_with_stations
```
