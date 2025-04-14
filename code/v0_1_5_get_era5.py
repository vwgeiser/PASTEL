import cdsapi

years = ["1992", "1993", "1994", "1995", "1996", "1997", "1998",
         "1999", "2000", "2001", "2002", "2003", "2004", "2005",
         "2006", "2007", "2008", "2009", "2010", "2011", "2012",
         "2013", "2014", "2015", "2016", "2017", "2018", "2019",
         "2020", "2021", "2022",] # 1991 already downloaded as test data.

for year in years:
    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "geopotential",
            "relative_humidity",
            "specific_humidity",
            "temperature"
        ],
        "pressure_level": [
            "250", "300", "350",
            "400", "450", "500",
            "550", "600", "650",
            "700", "750", "775",
            "800", "825", "850",
            "875", "900", "925",
            "950", "975", "1000"
        ],
        "year": [year],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()