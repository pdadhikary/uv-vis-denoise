import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def exp_decay(x, a, b, c):
    """
    a = amplitude (initial value scale)
    b = decay rate constant
    c = offset / asymptotic horizontal baseline
    """
    return a * np.exp(-b * x) + c


def main():
    st.set_page_config(layout="wide")
    st.header("UV-Vis Denoise")
    st.write(
        "Apply signal processing techniques to denoise ultraviolet-visible spectroscopy data"
    )

    file = st.file_uploader(label="Upload UV-Vis Excel File")

    if not file:
        st.write(
            "Please upload a UV-Vis Excel file to continue.\n\n"
            "The UV-Vis data should be in columnar format where the first column "
            "is Wavelength and subsequent columns contain Absorbance data measured "
            "at that specific wavelength. Below is an example of what this program "
            "expects. \n\n"
            "**Note: the column headers are arbitrary, you can name them anything you wish.**"
        )

        example_df = pd.DataFrame(
            {
                "wavelength": np.arange(450, 530),
                "Series 1": np.linspace(0.1, 0.6, 80),
                "Series 2": np.linspace(0.05, 0.3, 80),
                "Series N": np.linspace(0.5, 1.4, 80),
            }
        )

        st.dataframe(data=example_df)

    else:
        df = pd.read_excel(file)
        df = df.dropna(axis=1)
        wave_col = df.columns[0]
        cols = df.columns[1:]
        min_value, max_value = df[wave_col].min(), df[wave_col].max()

        a1, a2 = st.columns(2)

        with a1:
            start_wavelength, end_wavelength = st.slider(
                label="Wave Length Range",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value),
                step=1,
            )

            window_length = st.slider(
                label="Window Length",
                min_value=3,
                max_value=100,
                step=1,
                value=20,
            )

        with a2:
            polyorder = st.slider(
                label="Polynomial Order", min_value=1, max_value=3, step=1
            )

            band = st.slider(
                label="Spectral Band",
                min_value=start_wavelength,
                max_value=end_wavelength,
                step=1,
            )

        df = df[(df[wave_col] >= start_wavelength) & (df[wave_col] <= end_wavelength)]

        df_denoise = df.copy()

        for col in cols:
            df_denoise[col] = savgol_filter(
                x=df[col],
                window_length=window_length,
                polyorder=polyorder,
            )

        show_real = st.checkbox(label="Show Real Data", value=True)

        band_df = df_denoise.loc[df[wave_col] == band]
        band_values = band_df.values[0][1:]
        band_df = pd.DataFrame(
            {"Time": np.arange(len(band_values)), "Absorbance": band_values}
        )

        k1, k2 = st.columns(2)
        with k1:
            st.subheader("Denoised Adsorption Spectrum")
            st.dataframe(data=df_denoise, height="stretch")
        with k2:
            fig = go.Figure()
            fig.update_layout({"uirevision": "foo"}, overwrite=True)
            for col in cols:
                fig.add_trace(go.Scatter(x=df[wave_col], y=df_denoise[col], name=col))
                if show_real:
                    fig.add_trace(
                        go.Scatter(
                            x=df[wave_col], y=df[col], mode="markers", showlegend=False
                        )
                    )
                fig.update_layout(
                    title={"text": "Adsorption Spectrum", "font": {"size": 24}},
                    xaxis_title={"text": "Wavelength", "font": {"size": 18}},
                    yaxis_title={"text": "Absorbance", "font": {"size": 18}},
                    legend_font_size=18,
                    height=700,
                    legend={
                        "x": 0.85,
                        "y": 0.95,
                        "xanchor": "left",
                        "yanchor": "top",
                    },
                )
            st.plotly_chart(fig, key="filtered_chart")

        k3, k4 = st.columns(2)
        with k3:
            st.subheader("Adsorption and Concentration over Time Data")
            t1, t2 = st.columns(2)
            with t1:
                epsilon = st.number_input(
                    label="Molar Absorvity (Epsilon)",
                    min_value=0.0,
                    value=100.0,
                    step=0.5,
                )
            with t2:
                path_length = st.number_input(
                    label="Path Length (cm)", min_value=0.0, value=1.0, step=0.1
                )

            band_df["Concentration"] = band_df["Absorbance"] / (epsilon * path_length)

            st.write(
                "The table below is *EDITABLE*, you may use want to edit the Time columnn."
            )
            edited_band_df = st.data_editor(data=band_df, height="stretch")

        with k4:
            selected_col = st.pills(
                label="Plot Type",
                options=["Absorbance", "Concentration"],
                required=True,
                selection_mode="single",
                default="Absorbance",
            )
            popt, _ = curve_fit(
                exp_decay, edited_band_df["Time"], edited_band_df[selected_col]
            )
            fitted_x = np.linspace(
                edited_band_df["Time"].min(), edited_band_df["Time"].max(), 50
            )
            fitted_line = exp_decay(fitted_x, *popt)
            fig = go.Figure()
            fig.update_layout({"uirevision": "foo"}, overwrite=True)
            fig.add_trace(
                go.Scatter(
                    x=edited_band_df["Time"],
                    y=edited_band_df[selected_col],
                    mode="markers",
                    name=f"Observed {selected_col}",
                    marker={"size": 10},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fitted_x,
                    y=fitted_line,
                    mode="lines",
                    showlegend=False,
                )
            )
            fig.update_layout(
                title={
                    "text": f"{selected_col} vs Time (At Wavelength={band}nm)",
                    "font": {"size": 24},
                },
                xaxis_title={"text": "Time", "font": {"size": 18}},
                yaxis_title={"text": selected_col, "font": {"size": 18}},
                legend_font_size=18,
                height=700,
                legend={
                    "x": 0.85,
                    "y": 0.95,
                    "xanchor": "left",
                    "yanchor": "top",
                },
            )
            st.plotly_chart(fig, key="fitted_chart")


if __name__ == "__main__":
    main()
