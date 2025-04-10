# 📅 Weather Data

Several (and more coming) weather station format can be read and transformed to `DataFrame`.

## ECA dataset

From the [European Climate Assessment & Dataset project](https://www.ecad.eu/) at this [link for zip](https://www.ecad.eu/dailydata/predefinedseries.php) of all stations per variables and at this [link for custom](https://www.ecad.eu/dailydata/customquery.php) **manual** query.
I asked them about an API to extract directly a specific file automatically, but they answer it is not currently available.
I tried [unzip-http](https://github.com/saulpw/unzip-http) but could not get it working with ECA website[^1]

[^1]: I don't remember exactly in fact.

````@example data
using StochasticWeatherGenerators, DataFrames, Dates
collect_data_ECA(33, Date(1956), Date(2019, 12, 31), "https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/ECA_blend_rr/RR_", portion_valid_data=1, skipto=22, header=21, url=true)[1:10,:]
````

```@docs
collect_data_ECA
```

## [Météo France](@id DataMeteofrance)

Météo France do have a version of this data and it is accessible through an API on the website [Data.Gouv.fr](https://www.data.gouv.fr/en/datasets/).
This package provides a simple command to extract the data of one station (given its STAtionID) from the API.

````@example data
collect_data_MeteoFrance(49215002)[1:10,:]
````

```@docs
collect_data_MeteoFrance 
download_data_MeteoFrance 
```

## INRAE

The INRAE CLIMATIK platform [delannoy2022climatik](@cite) ([https://agroclim.inrae.fr/climatik/](https://agroclim.inrae.fr/climatik/), in French) managed by the AgroClim laboratory of Avignon, France has weather stations. However, their API is not open access.

```@docs
collect_data_INRAE
```

## Others

## Data manipulation

```@docs
clean_data 
select_in_range_df
shortname
```

## References

```@bibliography
Pages = ["data.md"]
```
