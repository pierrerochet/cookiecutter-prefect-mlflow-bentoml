import altair as alt


def train_test_chart(df):

    base = alt.Chart(df)

    bars = base.mark_bar().encode(
        x="count(target):Q",
        y=alt.Y("target:N", sort="-x"),
        color="group:N",
    )

    nb_detail = base.mark_text(dx=-15, color="white").encode(
        x=alt.X("count(target):Q", stack="zero"),
        y=alt.Y("target:N", sort="-x"),
        detail="group:N",
        text="count(target):Q",
    )

    nb_total = base.mark_text(dx=15, color="black").encode(
        x=alt.X("count(target):Q", stack="zero"),
        y=alt.Y("target:N", sort="-x"),
        text="count(target):Q",
    )

    chart = (
        (bars + nb_detail + nb_total)
        .properties(title="Data distribution")
        .configure_scale(bandPaddingOuter=1)
    )
    return chart


def count_bars(df):

    base = alt.Chart(df)

    bars = (
        base.mark_bar()
        .encode(
            x="count(target)",
            y=alt.Y("target:N", sort="-x"),
            color="target:N",
        )
        .properties(title="Category distribution")
    )

    text = base.mark_text(dx=-15, color="white").encode(
        x=alt.X("count(target):Q", stack="zero"),
        y=alt.Y("target:N", sort="-x"),
        text="count(target):Q",
    )

    chart = bars + text
    return chart
