# ruff: noqa
"""Public U.S. market time-series app entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "fintools").is_dir()),
    Path(__file__).resolve().parents[3],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fins2026.week2.app.app_config import (  # noqa: E402
    APP_TITLE,
    DATA_MODE_OPTIONS,
    DEFAULT_SAMPLE_PERIOD,
    DEFAULT_SERIES,
    SAMPLE_PERIODS,
    SERIES_LABELS,
    SERIES_NOTES,
)
from fins2026.week2.app.app_data import (  # noqa: E402
    apply_sample_period,
    load_market_data,
    source_status_text,
)
from fins2026.week2.code.market_window import WEEK2_FRED_WINDOW_YEARS  # noqa: E402
from fintools.apps import (  # noqa: E402
    add_nber_recession_vrects,
    apply_app_plotly_theme,
    configure_page,
    render_csv_download,
    render_data_health,
    render_display_table,
)


def series_figure(series, *, label: str) -> go.Figure:
    """Build a simple single-series time-series chart for the intro app."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=label,
            line={"color": "#355C7D", "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>" + label + ": %{y:,.2f}<extra></extra>",
        )
    )
    add_nber_recession_vrects(fig, start=series.index.min(), end=series.index.max(), opacity=0.24)
    apply_app_plotly_theme(
        fig,
        yaxis_title=label,
        height=460,
        range_selector=True,
    )
    return fig


def main() -> None:
    """Render the public U.S. month-end market and macro app."""

    import streamlit as st

    configure_page(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(
        "Track U.S. rates, labour, activity, policy, volatility, and equity-market "
        "measures from a unified month-end panel built from public FRED data."
    )

    with st.sidebar:
        st.header("Controls")
        data_mode = st.radio("Data source", DATA_MODE_OPTIONS, index=0)

    frame, active_data_mode, warning, loaded_at_utc = load_market_data(data_mode)
    if warning:
        st.warning(warning)

    sample_period = st.segmented_control(
        "Sample period",
        options=list(SAMPLE_PERIODS),
        default=DEFAULT_SAMPLE_PERIOD,
    )
    series_id = st.selectbox(
        "Series",
        list(SERIES_LABELS),
        index=list(SERIES_LABELS).index(DEFAULT_SERIES),
        format_func=lambda item: SERIES_LABELS[item],
    )

    sample_data = apply_sample_period(frame, sample_period)
    selected = sample_data[[series_id]].dropna()

    render_data_health(selected, source=active_data_mode)
    st.caption(
        "The chart view uses a unified month-end panel that samples daily market "
        "series at month-end, aligns monthly macro releases to the same calendar "
        f"endpoint, and standardizes the comparison window to the latest "
        f"{WEEK2_FRED_WINDOW_YEARS} years."
    )
    st.caption(
        source_status_text(
            frame,
            series=selected[series_id],
            series_label=SERIES_LABELS[series_id],
            active_data_mode=active_data_mode,
            loaded_at_utc=loaded_at_utc,
            warning=warning,
        )
    )

    if selected.empty:
        st.info("No observations are available for the selected series and sample window.")
        return

    st.subheader(SERIES_LABELS[series_id])
    if series_id in SERIES_NOTES:
        st.caption(SERIES_NOTES[series_id])
    st.plotly_chart(
        series_figure(selected[series_id], label=SERIES_LABELS[series_id]),
        use_container_width=True,
    )

    st.subheader("Displayed data")
    displayed = render_display_table(
        selected.tail(30),
        labels={series_id: SERIES_LABELS[series_id]},
    )
    render_csv_download(
        displayed,
        label="Download displayed data",
        file_name=f"week2_{series_id.lower()}_{sample_period.lower()}.csv",
        key=f"download_{series_id}_{sample_period}",
    )


if __name__ == "__main__":
    main()
