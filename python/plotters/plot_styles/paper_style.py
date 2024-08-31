from python.plotters.plot_styles.plot_style import (
    PlotStyle,
    TITLE_ANNOTATIONS,
)

paper_style = PlotStyle(
    style="paperStyle_",
    binning="auto",
    y_scale=1.16,
    fig={
        "size": (8, 8),
        "subplot_ratio": [7, 3],
        "sharex": True,
    },
    subplot={
        "left": 0.08,
        "right": 0.97,
        "bottom": 0.07,
        "top": 0.96,
        "hspace": 0.03,
    },
    legend={
        "loc": "upper right",
        "fontsize": 10,
    },
    colors={
        "data": "black",
        "mc": "cornflowerblue",
        "syst": "red",
    },
    line_styles={
        "data": "",
        "mc": "",
        "syst": "",
        "ratio": "",
    },
    labels={
        "x_axis": {
            "label": "M$_{ee}$ [GeV]",
            "fontsize": 14,
            "ha": "right",
        },
        "data": "Data",
        "mc": "MC",
        "syst": "MC stat. $\oplus$ syst. unc.",
        "ratio": "Data / MC",
    },
    error_bar_style="steps-mid",
    sci_notation_offset=(-0.065, 0.5),
    annotations={
        "lumi": {
            "annot": r"XX.X fb$^{-1}$ (13 TeV) 20XX",
            "xy": (1, 1.0),
            "xycoords": "axes fraction",
            "ha": "right",
            "va": "bottom",
            "fontsize": 14,
        },
        "cms_tag": {
            "annot": r"$\textbf{CMS}$ $\textit{Preliminary}$",
            "xy": (0, 1.0),
            "xycoords": "axes fraction",
            "ha": "left",
            "va": "bottom",
            "fontsize": 14,
        },
        "plot_title": {
            "annot": TITLE_ANNOTATIONS,
            "xy": (0.1, 0.9),
            "xycoords": "axes fraction",
            "ha": "left",
            "va": "top",
            "fontsize": 18,
            "weight": "bold",
        },
    },
)
