# ruff: noqa
"""Streamlit layout and controls for the Week 5 client-facing crypto fund app."""

from __future__ import annotations

import pandas as pd

from fins2026.week5.app.app_config import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_DESIGN_FREQUENCY,
    DEFAULT_DESIGN_INITIAL_WINDOW,
    DEFAULT_DESIGN_WINDOW_RULE,
    DEFAULT_INVESTMENT_AMOUNT,
    DEFAULT_PORTFOLIO_KEY,
    DEFAULT_SAMPLE_PERIOD,
    DEFAULT_VIEW,
    DESIGN_FREQUENCY_OPTIONS,
    DESIGN_INITIAL_WINDOW_OPTIONS,
    DESIGN_WINDOW_RULE_OPTIONS,
    METHOD_MAPPING,
    METHOD_NOTES,
    PORTFOLIO_DESCRIPTIONS,
    PORTFOLIO_METHOD_LABELS,
    PORTFOLIO_OPTIONS,
    SAMPLE_PERIOD_OPTIONS,
    VIEW_OPTIONS,
)
from fins2026.week5.app.app_data import (
    build_design_scenario,
    build_published_scenario,
    load_week5_app_bundle,
)
from fins2026.week5.app.app_insights import (
    allocation_figure,
    allocation_table,
    compact_table_height,
    concentration_cards,
    concentration_table,
    design_metric_cards,
    drawdown_figure,
    frontier_figure,
    growth_figure,
    holdings_snapshot_figure,
    latest_btc_eth_exposure_table,
    live_weight_table,
    methodology_table,
    performance_comparison_table,
    risk_contribution_figure,
    selected_fund_metric_cards,
    target_weight_table,
    trailing_return_snapshot_figure,
    trailing_risk_comparison_table,
    trailing_risk_table,
    turnover_change_figure,
    turnover_table,
)
from fins2026.week5.code.stage4_app import (
    APP_PUBLISHED_ESTIMATION_FREQUENCY,
    APP_PUBLISHED_INITIAL_WINDOW,
    APP_PUBLISHED_WINDOW_RULE,
    build_display_window_analysis,
    build_investment_allocation,
    published_fund_name,
    source_status_text,
)
from fintools.apps import (
    active_tab_label,
    configure_page,
    lazy_tabs,
    query_choice,
    query_int,
    render_compact_metric_strip,
    render_csv_download,
    render_data_health,
    render_display_table,
    sync_query_params,
    tab_is_open,
)


def _initialize_state(st) -> None:
    st.session_state.setdefault(
        "week5_portfolio_key",
        query_choice("fund", list(PORTFOLIO_OPTIONS), default=DEFAULT_PORTFOLIO_KEY),
    )
    st.session_state.setdefault(
        "week5_sample_period",
        query_choice("sample", list(SAMPLE_PERIOD_OPTIONS), default=DEFAULT_SAMPLE_PERIOD),
    )
    st.session_state.setdefault(
        "week5_investment_amount",
        query_int(
            "amount",
            default=DEFAULT_INVESTMENT_AMOUNT,
            minimum=1_000,
            maximum=10_000_000,
            step=500,
        ),
    )
    st.session_state.setdefault(
        "week5_design_frequency",
        query_choice(
            "design_freq",
            list(DESIGN_FREQUENCY_OPTIONS),
            default=DEFAULT_DESIGN_FREQUENCY,
        ),
    )
    st.session_state.setdefault(
        "week5_design_initial_window",
        query_int(
            "design_window",
            default=DEFAULT_DESIGN_INITIAL_WINDOW,
            minimum=min(DESIGN_INITIAL_WINDOW_OPTIONS),
            maximum=max(DESIGN_INITIAL_WINDOW_OPTIONS),
            step=1,
        ),
    )
    if st.session_state["week5_design_initial_window"] not in DESIGN_INITIAL_WINDOW_OPTIONS:
        st.session_state["week5_design_initial_window"] = DEFAULT_DESIGN_INITIAL_WINDOW
    st.session_state.setdefault(
        "week5_design_window_rule",
        query_choice(
            "design_rule",
            list(DESIGN_WINDOW_RULE_OPTIONS),
            default=DEFAULT_DESIGN_WINDOW_RULE,
        ),
    )


def _render_sidebar_controls(st) -> tuple[str, str, int, str, int, str]:
    with st.sidebar:
        st.header("Published lineup")
        portfolio_key = st.selectbox(
            "Published fund",
            list(PORTFOLIO_OPTIONS),
            format_func=lambda key: PORTFOLIO_OPTIONS[key],
            key="week5_portfolio_key",
        )
        sample_period = (
            st.segmented_control(
                "Chart window",
                list(SAMPLE_PERIOD_OPTIONS),
                key="week5_sample_period",
            )
            or st.session_state["week5_sample_period"]
        )
        investment_amount = int(
            st.number_input(
                "Illustrative investment (USD)",
                min_value=1_000,
                step=500,
                key="week5_investment_amount",
            )
        )

        with st.expander("Portfolio design controls", expanded=False):
            st.caption(
                "These settings update only the Portfolio Design view. The "
                "published lineup remains monthly, 365-day, and expanding."
            )
            with st.form("week5_design_controls"):
                design_frequency = st.radio(
                    "Rebalance schedule",
                    list(DESIGN_FREQUENCY_OPTIONS),
                    format_func=lambda key: DESIGN_FREQUENCY_OPTIONS[key],
                    key="week5_design_frequency",
                )
                design_initial_window = st.selectbox(
                    "Training window",
                    DESIGN_INITIAL_WINDOW_OPTIONS,
                    key="week5_design_initial_window",
                )
                design_window_rule = st.radio(
                    "Window rule",
                    list(DESIGN_WINDOW_RULE_OPTIONS),
                    format_func=lambda key: DESIGN_WINDOW_RULE_OPTIONS[key],
                    key="week5_design_window_rule",
                )
                st.form_submit_button("Update design view")
    return (
        portfolio_key,
        sample_period,
        investment_amount,
        design_frequency,
        int(design_initial_window),
        design_window_rule,
    )


def _render_overview_tab(
    st,
    *,
    published_window,
    published_scenario,
) -> None:
    st.subheader("Published fund lineup")
    st.caption(
        "Historical comparisons use only information that would have been available "
        "at each rebalance date."
    )
    st.plotly_chart(growth_figure(published_window), width="stretch")
    st.plotly_chart(drawdown_figure(published_window), width="stretch")
    st.subheader("Fund comparison")
    metric_table = performance_comparison_table(published_window.metrics)
    render_display_table(
        metric_table,
        reset_index=False,
        height=compact_table_height(metric_table, max_height=420),
    )
    exposure_table = latest_btc_eth_exposure_table(published_scenario.btc_eth_exposure)
    st.subheader("Current majors exposure")
    render_display_table(
        exposure_table,
        reset_index=False,
        height=compact_table_height(exposure_table, max_height=280),
    )


def _render_fund_details_tab(
    st,
    *,
    portfolio_key: str,
    published_scenario,
) -> None:
    st.subheader(published_fund_name(portfolio_key))
    st.markdown(PORTFOLIO_DESCRIPTIONS[portfolio_key])
    render_compact_metric_strip(
        concentration_cards(
            published_scenario.concentration_snapshot,
            published_scenario.turnover_snapshot,
            published_scenario.btc_eth_exposure,
            portfolio_key=portfolio_key,
        ),
        columns=3,
    )

    holdings_cols = st.columns(2)
    with holdings_cols[0]:
        st.plotly_chart(
            holdings_snapshot_figure(
                published_scenario.latest_live_weights,
                portfolio_key=portfolio_key,
                weight_column="weight",
                title="Current live exposure",
            ),
            width="stretch",
        )
    with holdings_cols[1]:
        st.plotly_chart(
            holdings_snapshot_figure(
                published_scenario.latest_target_weights,
                portfolio_key=portfolio_key,
                weight_column="latest_weight",
                title="Latest target exposure",
            ),
            width="stretch",
        )

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.plotly_chart(
            trailing_return_snapshot_figure(
                published_scenario.trailing_returns,
                portfolio_key=portfolio_key,
            ),
            width="stretch",
        )
    with detail_cols[1]:
        st.plotly_chart(
            turnover_change_figure(
                published_scenario.latest_target_weights,
                portfolio_key=portfolio_key,
            ),
            width="stretch",
        )

    risk_cols = st.columns(2)
    with risk_cols[0]:
        st.plotly_chart(
            risk_contribution_figure(
                published_scenario.risk_contributions,
                portfolio_key=portfolio_key,
            ),
            width="stretch",
        )
    with risk_cols[1]:
        risk_table = trailing_risk_table(
            published_scenario.trailing_risk,
            portfolio_key=portfolio_key,
        )
        st.subheader("Current risk snapshot")
        render_display_table(
            risk_table,
            reset_index=False,
            height=compact_table_height(risk_table, max_height=240),
        )

    table_cols = st.columns(2)
    with table_cols[0]:
        st.subheader("Current live holdings")
        live_table = live_weight_table(
            published_scenario.latest_live_weights,
            portfolio_key=portfolio_key,
        )
        render_display_table(
            live_table,
            reset_index=False,
            height=compact_table_height(live_table, max_height=420),
        )
    with table_cols[1]:
        st.subheader("Latest target weights")
        target_table = target_weight_table(
            published_scenario.latest_target_weights,
            portfolio_key=portfolio_key,
        )
        render_display_table(
            target_table,
            reset_index=False,
            height=compact_table_height(target_table, max_height=420),
        )


def _render_invest_tab(
    st,
    *,
    portfolio_key: str,
    investment_amount: int,
    published_scenario,
) -> None:
    st.subheader("Illustrative allocation")
    st.caption(
        "The table and chart below translate the selected investment amount into "
        "today's live coin mix for the published fund."
    )
    allocation = build_investment_allocation(
        published_scenario.latest_live_weights,
        portfolio_key=portfolio_key,
        amount_usd=float(investment_amount),
    )
    allocation_fig = allocation_figure(allocation, portfolio_key=portfolio_key)
    allocation_tbl = allocation_table(allocation)

    alloc_cols = st.columns([1.35, 1.0])
    with alloc_cols[0]:
        st.plotly_chart(allocation_fig, width="stretch")
    with alloc_cols[1]:
        render_display_table(
            allocation_tbl,
            reset_index=False,
            height=compact_table_height(allocation_tbl, max_height=440),
        )
        render_csv_download(
            allocation_tbl,
            label="Download allocation",
            file_name=f"{portfolio_key}_illustrative_allocation.csv",
            key=f"download_allocation_{portfolio_key}",
        )
    st.info(
        "This allocation view is an illustration of the current model portfolio. "
        "It is not trade execution, personal advice, or a promise of future returns."
    )


def _render_design_tab(
    st,
    *,
    feature_panel: pd.DataFrame,
    sample_period: str,
    portfolio_key: str,
    design_frequency: str,
    design_initial_window: int,
    design_window_rule: str,
) -> None:
    st.subheader("Portfolio design comparison")
    st.caption(
        "This view compares alternative rebalance settings while keeping the same "
        "five published fund definitions."
    )
    with st.spinner("Refreshing the alternative design view..."):
        design_scenario = build_design_scenario(
            feature_panel,
            estimation_frequency=design_frequency,
            initial_window=design_initial_window,
            window_rule=design_window_rule,
        )
    render_compact_metric_strip(
        design_metric_cards(design_scenario, portfolio_key=portfolio_key),
        columns=3,
    )
    if (
        design_frequency == APP_PUBLISHED_ESTIMATION_FREQUENCY
        and design_initial_window == APP_PUBLISHED_INITIAL_WINDOW
        and design_window_rule == APP_PUBLISHED_WINDOW_RULE
    ):
        st.info("These settings match the official published lineup.")
    else:
        st.info(
            "These settings are a design comparison only. The published lineup "
            "remains monthly, 365-day, and expanding."
        )

    display_window = build_display_window_analysis(
        design_scenario,
        sample_period=sample_period,
    )
    st.plotly_chart(growth_figure(display_window), width="stretch")
    design_metric_table = performance_comparison_table(display_window.metrics)
    render_display_table(
        design_metric_table,
        reset_index=False,
        height=compact_table_height(design_metric_table, max_height=420),
    )
    design_cols = st.columns(2)
    with design_cols[0]:
        st.plotly_chart(
            holdings_snapshot_figure(
                design_scenario.latest_live_weights,
                portfolio_key=portfolio_key,
                weight_column="weight",
                title="Current live exposure",
            ),
            width="stretch",
        )
    with design_cols[1]:
        st.plotly_chart(
            frontier_figure(display_window, active_portfolio_key=portfolio_key),
            width="stretch",
        )


def _render_data_tab(st, *, bundle, published_scenario) -> None:
    st.subheader("Data & downloads")
    table_choice = (
        st.segmented_control(
            "Dataset",
            [
                "Performance series",
                "Current live holdings",
                "Latest target weights",
                "Concentration",
                "Trailing risk",
                "Turnover",
            ],
            key="week5_data_table_choice",
        )
        or "Performance series"
    )
    if table_choice == "Performance series":
        frame = published_scenario.portfolio_returns.copy()
        frame["return_date"] = pd.to_datetime(frame["return_date"]).dt.strftime("%Y-%m-%d")
        frame["formation_date"] = pd.to_datetime(frame["formation_date"]).dt.strftime("%Y-%m-%d")
        for portfolio_col in PORTFOLIO_OPTIONS:
            frame[portfolio_col] = frame[portfolio_col].astype(float) * 100.0
        labels = {
            "return_date": "Performance date",
            "formation_date": "Latest rebalance date",
            **{key: f"{PORTFOLIO_OPTIONS[key]} daily return (%)" for key in PORTFOLIO_OPTIONS},
        }
        file_name = "week5_published_portfolio_returns.csv"
    elif table_choice == "Current live holdings":
        frame = published_scenario.latest_live_weights.copy()
        frame["return_date"] = pd.to_datetime(frame["return_date"]).dt.strftime("%Y-%m-%d")
        frame["formation_date"] = pd.to_datetime(frame["formation_date"]).dt.strftime("%Y-%m-%d")
        frame["Fund"] = frame["portfolio_key"].map(PORTFOLIO_OPTIONS)
        frame["weight"] = frame["weight"].astype(float) * 100.0
        labels = {
            "return_date": "Performance date",
            "formation_date": "Latest rebalance date",
            "ticker": "Coin",
            "weight": "Live weight (%)",
        }
        frame = frame.loc[:, ["Fund", "ticker", "weight", "return_date", "formation_date"]]
        file_name = "week5_published_live_weights.csv"
    elif table_choice == "Latest target weights":
        frame = published_scenario.latest_target_weights.copy()
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.strftime("%Y-%m-%d")
        frame["previous_decision_date"] = pd.to_datetime(
            frame["previous_decision_date"]
        ).dt.strftime("%Y-%m-%d")
        frame["Fund"] = frame["portfolio_key"].map(PORTFOLIO_OPTIONS)
        frame["latest_weight"] = frame["latest_weight"].astype(float) * 100.0
        frame["previous_weight"] = frame["previous_weight"].astype(float) * 100.0
        frame["weight_change"] = frame["weight_change"].astype(float) * 100.0
        labels = {
            "decision_date": "Latest rebalance date",
            "previous_decision_date": "Previous rebalance date",
            "ticker": "Coin",
            "latest_weight": "Target weight (%)",
            "previous_weight": "Previous target (%)",
            "weight_change": "Weight change (%)",
        }
        frame = frame.loc[
            :,
            [
                "Fund",
                "ticker",
                "latest_weight",
                "previous_weight",
                "weight_change",
                "decision_date",
                "previous_decision_date",
            ],
        ]
        file_name = "week5_published_target_weights.csv"
    elif table_choice == "Concentration":
        frame = concentration_table(published_scenario.concentration_snapshot)
        labels = None
        file_name = "week5_concentration_snapshot.csv"
    elif table_choice == "Trailing risk":
        frame = trailing_risk_comparison_table(published_scenario.trailing_risk)
        labels = None
        file_name = "week5_trailing_risk_snapshot.csv"
    else:
        frame = turnover_table(published_scenario.turnover_snapshot)
        labels = None
        file_name = "week5_turnover_snapshot.csv"

    displayed = render_display_table(
        frame,
        labels=labels,
        reset_index=False,
        height=compact_table_height(frame, max_height=560),
    )
    render_csv_download(
        displayed,
        label="Download selected table",
        file_name=file_name,
        key=f"download_{table_choice.lower().replace(' ', '_').replace('&', 'and')}",
    )
    st.subheader("Current data health")
    render_data_health(
        bundle.feature_panel,
        source="Published crypto data",
        date_column="date",
        value_columns=["adjClose", "ret", "rfr"],
    )


def _render_methodology_tab(st) -> None:
    st.subheader("Methodology")
    st.markdown(
        "The published lineup follows five long-only rules that rebalance using "
        "historical information only. The table below maps each visible fund name "
        "to the underlying portfolio-construction method."
    )
    method_table = methodology_table(METHOD_MAPPING)
    render_display_table(
        method_table,
        reset_index=False,
        height=compact_table_height(method_table, max_height=360),
    )
    st.markdown("Mean-variance portfolio:")
    st.latex(
        r"\max_{w}\ \frac{\mathbb{E}[r_p-r_f]}{\sigma_p}\ \ \text{s.t.}\ \sum_i w_i=1,\ w_i\ge 0"
    )
    st.markdown("Minimum-variance portfolio:")
    st.latex(r"\min_{w}\ w^{\top}\Sigma w\ \ \text{s.t.}\ \sum_i w_i=1,\ w_i\ge 0")
    st.markdown("Risk-parity portfolio:")
    st.latex(r"w_i(\Sigma w)_i \approx \frac{1}{N} w^{\top}\Sigma w")
    st.markdown("Mean-CVaR portfolio:")
    st.latex(
        r"\max_{w}\ \frac{\mathbb{E}[r_p-r_f]}{\mathrm{CVaR}_{95\%}(-r_p)}"
        r"\ \ \text{s.t.}\ \sum_i w_i=1,\ w_i\ge 0"
    )
    st.markdown(
        "Daily cash rates come from the Kenneth French research factors file. "
        "Because crypto trades every day while the cash-rate series is business-day "
        "only, the app forward-fills the most recent available cash rate across "
        "weekends and holidays before computing excess returns."
    )
    st.markdown("Published fund notes:")
    for portfolio_key, note in METHOD_NOTES.items():
        st.markdown(
            f"- **{PORTFOLIO_OPTIONS[portfolio_key]}** "
            f"(`{PORTFOLIO_METHOD_LABELS[portfolio_key]}`): {note}"
        )


def main() -> None:
    """Run the Week 5 client-facing crypto fund app."""

    st = configure_page(APP_TITLE)
    _initialize_state(st)
    (
        portfolio_key,
        sample_period,
        investment_amount,
        design_frequency,
        design_initial_window,
        design_window_rule,
    ) = _render_sidebar_controls(st)

    bundle, active_source, warning, loaded_at_utc = load_week5_app_bundle()
    published_scenario = build_published_scenario(bundle.feature_panel)
    published_window = build_display_window_analysis(
        published_scenario,
        sample_period=sample_period,
    )

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    if warning:
        st.warning(warning)
    st.caption(
        source_status_text(
            bundle,
            active_source=active_source,
            loaded_at_utc=loaded_at_utc,
            warning=warning,
        )
    )
    render_data_health(
        bundle.feature_panel,
        source=active_source,
        date_column="date",
        value_columns=["adjClose", "ret", "rfr"],
    )
    render_compact_metric_strip(
        selected_fund_metric_cards(
            published_scenario,
            portfolio_key=portfolio_key,
        ),
        columns=3,
    )

    view_default = query_choice("view", VIEW_OPTIONS, default=DEFAULT_VIEW)
    tabs = lazy_tabs(VIEW_OPTIONS, default=view_default, key="week5_app_view")
    active_view = active_tab_label(VIEW_OPTIONS, tabs, default=view_default)
    sync_query_params(
        view=active_view,
        fund=portfolio_key,
        sample=sample_period,
        amount=investment_amount,
        design_freq=design_frequency,
        design_window=design_initial_window,
        design_rule=design_window_rule,
    )
    (
        tab_overview,
        tab_details,
        tab_invest,
        tab_design,
        tab_data,
        tab_method,
    ) = tabs

    if tab_is_open(tab_overview, fallback=active_view == "Overview"):
        with tab_overview:
            _render_overview_tab(
                st,
                published_window=published_window,
                published_scenario=published_scenario,
            )

    if tab_is_open(tab_details, fallback=active_view == "Fund Details"):
        with tab_details:
            _render_fund_details_tab(
                st,
                portfolio_key=portfolio_key,
                published_scenario=published_scenario,
            )

    if tab_is_open(tab_invest, fallback=active_view == "Invest"):
        with tab_invest:
            _render_invest_tab(
                st,
                portfolio_key=portfolio_key,
                investment_amount=investment_amount,
                published_scenario=published_scenario,
            )

    if tab_is_open(tab_design, fallback=active_view == "Portfolio Design"):
        with tab_design:
            _render_design_tab(
                st,
                feature_panel=bundle.feature_panel,
                sample_period=sample_period,
                portfolio_key=portfolio_key,
                design_frequency=design_frequency,
                design_initial_window=design_initial_window,
                design_window_rule=design_window_rule,
            )

    if tab_is_open(tab_data, fallback=active_view == "Data & Downloads"):
        with tab_data:
            _render_data_tab(
                st,
                bundle=bundle,
                published_scenario=published_scenario,
            )

    if tab_is_open(tab_method, fallback=active_view == "Methodology"):
        with tab_method:
            _render_methodology_tab(st)
