import altair as alt


def cm_chart(df):
    base = alt.Chart(df).encode(x="true:O", y="pred:O")

    # Text layer with correlation labels
    # Colors are for easier readability
    text = base.mark_text().encode(
        text="value",
        color=alt.condition(
            alt.datum.value < df.value.mean(), alt.value("black"), alt.value("white")
        ),
    )

    # The correlation heatmap itself
    cor_plot = base.mark_rect().encode(color="value:Q")

    chart = (cor_plot + text).properties(
        title="Confusion matrix", width=300, height=300
    )
    return chart
