# IMAP Quicklook

This IMAP repository holds the code to generate quicklook plots: simple graphs that indicate the health of the different
IMAP instruments and general data collection abilities. These graphs will be utilized by the engineers, scientists, and
other individuals working on the IMAP teams ranging from software to science.

In order to produce the quicklook plots coded here you must:

1.) Initialize a `QuicklookGenerator` object taking in the file name you wish to plot.

2.) Call the method `two_dimensional_plot()` method on the `data_set` member.

*Note: If you are dealing with an instrument with multiple quicklook plots, you must enter a variable parameter to this function all to produce the desired image. Otherwise, you do not need a parameter.*

See below an example given for the MAG instrument:

```python
mag_data = MagQuicklookGenerator("imap_mag_l1a_norm-magi_20251017_v001.cdf")
mag_data.two_dimensional_plot("mag sensor co-ord")
```
Through the use of abstract classes, the `two_dimensal_plot` member will call the correct graphing function for the desired instrument.

This Git repository allows contributors to explore different xarray dataset variables being used to generate the
quicklook plots, the code behind the `QuicklookPlotGenerator` class, and to propose bug fixes and changes.

[IMAP Website](https://imap.princeton.edu/)

[IMAP Processing Documentation](https://imap-processing.readthedocs.io/en/latest/)

[Getting started](https://imap-processing.readthedocs.io/en/latest/development-guide/getting-started.html)

# Credits
[LASP (Laboratory of Atmospheric and Space Physics)](https://lasp.colorado.edu/)
