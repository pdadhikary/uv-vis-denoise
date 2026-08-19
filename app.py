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
        st.write("Please upload a UV-Vis Excel file to continue.")
    else:
        df = pd.read_excel(file)
        df = df.dropna(axis=1)

        cols = [col for col in df.columns if col.startswith("Title")]
        min_value, max_value = df["Wave"].min(), df["Wave"].max()

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

        df = df[(df["Wave"] >= start_wavelength) & (df["Wave"] <= end_wavelength)]

        df_denoise = df.copy()

        for col in cols:
            df_denoise[col] = savgol_filter(
                x=df[col],
                window_length=window_length,
                polyorder=polyorder,
            )

        show_real = st.checkbox(label="Show Real Data", value=True)

        band_df = df_denoise.loc[df["Wave"] == band]
        band_values = band_df.values[0][1:]
        band_df = pd.DataFrame(
            {"Time": np.arange(len(band_values)), "Absorbance": band_values}
        )

        k1, k2 = st.columns(2)
        with k1:
            st.subheader("Denoised Absorption Spectrum")
            st.dataframe(data=df_denoise, height="stretch")
        with k2:
            fig = go.Figure()
            fig.update_layout({"uirevision": "foo"}, overwrite=True)
            for col in cols:
                fig.add_trace(go.Scatter(x=df["Wave"], y=df_denoise[col], name=col))
                if show_real:
                    fig.add_trace(
                        go.Scatter(
                            x=df["Wave"], y=df[col], mode="markers", showlegend=False
                        )
                    )
                fig.update_layout(
                    title={"text": "Absorption Spectrum", "font": {"size": 24}},
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
            st.subheader("Absorption over Time Data")
            edited_band_df = st.data_editor(data=band_df, height="stretch")

        with k4:
            popt, _ = curve_fit(
                exp_decay, edited_band_df["Time"], edited_band_df["Absorbance"]
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
                    y=edited_band_df["Absorbance"],
                    mode="markers",
                    name="Observed Absorbance",
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
                    "text": f"Absorption over Time (At Wavelength={band}nm)",
                    "font": {"size": 24},
                },
                xaxis_title={"text": "Time", "font": {"size": 18}},
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
            st.plotly_chart(fig, key="fitted_chart")


if __name__ == "__main__":
    main()
