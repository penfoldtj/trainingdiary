import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from datetime import datetime, date, timedelta

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Training Diary", layout="wide")

DB_PATH = "training_diary.db"  # <-- your persistent file

# -----------------------------
# Utilities
# -----------------------------
def parse_duration_to_seconds(s: str) -> int:
    """Accept 'HH:MM:SS' or 'MM:SS' or minutes as number-like; returns total seconds."""
    if s is None or str(s).strip() == "":
        return 0
    s = str(s).strip()
    if s.isdigit():
        return int(float(s) * 60)  # treat plain number as minutes
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return int(h * 3600 + m * 60 + sec)
        elif len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return int(m * 60 + sec)
        else:
            # fallback: treat as minutes float
            return int(float(s) * 60)
    except Exception:
        return 0

def format_seconds(sec: int) -> str:
    """Format seconds as HH:MM:SS."""
    if sec is None:
        return "00:00:00"
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def calc_pace(duration_s: int, distance_km: float) -> str:
    """Return pace as mm:ss per km. If not computable, return '-'."""
    if duration_s and distance_km and distance_km > 0:
        sec_per_km = duration_s / distance_km
        m = int(sec_per_km // 60)
        s = int(sec_per_km % 60)
        return f"{m:02d}:{s:02d}/km"
    return "-"

# -----------------------------
# Database Helpers
# -----------------------------
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_date TEXT NOT NULL,      -- YYYY-MM-DD
          sport TEXT NOT NULL,             -- e.g., Run, Bike, Swim, S&C
          session_type TEXT,               -- e.g., Easy, Intervals, Long, Race, Tempo
          distance_km REAL,                -- km (can be 0/NULL for non-distance sessions)
          duration_s INTEGER,              -- total seconds
          rpe INTEGER,                     -- 1..10
          notes TEXT
        )
    """)
    conn.commit()
    return conn

def insert_entry(conn, session_date, sport, session_type, distance_km, duration_s, rpe, notes):
    conn.execute(
        "INSERT INTO entries (session_date, sport, session_type, distance_km, duration_s, rpe, notes) VALUES (?,?,?,?,?,?,?)",
        (session_date, sport, session_type, distance_km, duration_s, rpe, notes),
    )
    conn.commit()

def load_entries(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY session_date DESC, id DESC", conn)
    if not df.empty:
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    return df

def update_entry(conn, row):
    conn.execute(
        """UPDATE entries SET session_date=?, sport=?, session_type=?, distance_km=?, duration_s=?, rpe=?, notes=?
           WHERE id=?""",
        (
            row["session_date"].strftime("%Y-%m-%d") if isinstance(row["session_date"], date) else str(row["session_date"]),
            row["sport"],
            row.get("session_type"),
            float(row["distance_km"]) if pd.notna(row["distance_km"]) else None,
            int(row["duration_s"]) if pd.notna(row["duration_s"]) else None,
            int(row["rpe"]) if pd.notna(row["rpe"]) else None,
            row.get("notes"),
            int(row["id"]),
        ),
    )
    conn.commit()

def delete_entries(conn, ids):
    if not ids:
        return
    q_marks = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM entries WHERE id IN ({q_marks})", ids)
    conn.commit()

# -----------------------------
# Sidebar: Add Session
# -----------------------------
st.sidebar.header("Add Training Session")

with st.sidebar.form("add_form", clear_on_submit=True):
    d = st.date_input("Date", value=date.today())
    sport = st.selectbox("Sport", ["Run", "Bike", "Swim", "S&C", "Other"])
    session_type = st.selectbox("Type", ["Easy", "Intervals", "Tempo", "Long", "Race", "Recovery", "Other"])
    colA, colB = st.columns(2)
    with colA:
        distance_km = st.number_input("Distance (km)", min_value=0.0, step=0.1, value=0.0)
    with colB:
        duration_str = st.text_input("Duration (HH:MM:SS or MM:SS or minutes)", value="00:45:00")
    rpe = st.slider("RPE (1 easy - 10 max)", 1, 10, 6)
    notes = st.text_area("Notes", placeholder="Optional details…")
    submitted = st.form_submit_button("Add Session", type="primary")

conn = get_conn()

if submitted:
    duration_s = parse_duration_to_seconds(duration_str)
    insert_entry(
        conn,
        session_date=d.strftime("%Y-%m-%d"),
        sport=sport,
        session_type=session_type,
        distance_km=distance_km if distance_km > 0 else None,
        duration_s=duration_s if duration_s > 0 else None,
        rpe=rpe,
        notes=notes.strip() if notes else None,
    )
    st.sidebar.success("Session added ✔️")

# -----------------------------
# Load & Filters
# -----------------------------
df = load_entries(conn)

st.title("🏃‍♀️ Training Diary")

# Filters row
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2])
with col1:
    if df.empty:
        min_date, max_date = date.today(), date.today()
    else:
        min_date, max_date = df["session_date"].min(), df["session_date"].max()
    start_date = st.date_input("From", value=max(min_date, date.today() - timedelta(days=30)))
with col2:
    end_date = st.date_input("To", value=max_date)
with col3:
    sport_filter = st.multiselect("Sport", sorted(df["sport"].unique()) if not df.empty else [], default=None)
with col4:
    text_filter = st.text_input("Search Notes/Type", placeholder="e.g., hill, race, easy…")

mask = pd.Series([True] * len(df))
if not df.empty:
    mask &= df["session_date"].between(start_date, end_date)
    if sport_filter:
        mask &= df["sport"].isin(sport_filter)
    if text_filter.strip():
        txt = text_filter.strip().lower()
        mask &= (
            df["notes"].fillna("").str.lower().str.contains(txt)
            | df["session_type"].fillna("").str.lower().str.contains(txt)
        )

filtered = df[mask].copy() if not df.empty else df.copy()

# Derived columns for display
if not filtered.empty:
    filtered["Duration"] = filtered["duration_s"].apply(lambda x: format_seconds(int(x)) if pd.notna(x) else "")
    filtered["Pace"] = filtered.apply(
        lambda r: calc_pace(int(r["duration_s"]), float(r["distance_km"])) if pd.notna(r["duration_s"]) and pd.notna(r["distance_km"]) else "-",
        axis=1,
    )
    filtered = filtered.sort_values(["session_date", "id"], ascending=[False, False])

# -----------------------------
# KPI Bar
# -----------------------------
def secs_sum(series):
    s = int(series.dropna().sum()) if not series.empty else 0
    return format_seconds(s)

colA, colB, colC, colD = st.columns(4)
if filtered.empty:
    colA.metric("Sessions", 0)
    colB.metric("Distance (km)", "0.0")
    colC.metric("Time", "00:00:00")
    colD.metric("Avg RPE", "-")
else:
    sessions = len(filtered)
    dist = float(filtered["distance_km"].dropna().sum()) if "distance_km" in filtered else 0.0
    time_fmt = secs_sum(filtered["duration_s"]) if "duration_s" in filtered else "00:00:00"
    rpe_avg = filtered["rpe"].dropna().mean() if "rpe" in filtered else None
    colA.metric("Sessions", sessions)
    colB.metric("Distance (km)", f"{dist:.1f}")
    colC.metric("Time", time_fmt)
    colD.metric("Avg RPE", f"{rpe_avg:.1f}" if rpe_avg is not None else "-")

# -----------------------------
# Charts
# -----------------------------
if not filtered.empty:
    # Weekly totals (distance & time)
    chart_df = filtered.copy()
    chart_df["week"] = chart_df["session_date"].apply(
        lambda d: date.fromisocalendar(d.isocalendar().year, d.isocalendar().week, 1)  # Monday of ISO week
    )
    weekly = chart_df.groupby(["week"], as_index=False).agg(
        distance_km=("distance_km", "sum"),
        duration_s=("duration_s", "sum"),
        sessions=("id", "count"),
    )
    weekly["duration_h"] = weekly["duration_s"] / 3600.0

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(weekly, x="week", y="distance_km", title="Weekly Distance (km)")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.bar(weekly, x="week", y="duration_h", title="Weekly Time (hours)")
        st.plotly_chart(fig2, use_container_width=True)

    # RPE trend
    rpe_trend = filtered.dropna(subset=["rpe"]).copy()
    if not rpe_trend.empty:
        fig3 = px.scatter(rpe_trend, x="session_date", y="rpe", color="sport",
                          hover_data=["session_type", "notes"], title="RPE Over Time", trendline="lowess")
        st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Editor (inline edit & delete)
# -----------------------------
st.subheader("Log")

if filtered.empty:
    st.info("No sessions yet. Add your first one from the sidebar!")
else:
    # Show a light-weight table with editable fields (not all columns editable)
    show_cols = ["id", "session_date", "sport", "session_type", "distance_km", "Duration", "rpe", "Pace", "notes"]
    edited = st.data_editor(
        filtered[show_cols],
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "session_date": st.column_config.DateColumn("Date"),
            "distance_km": st.column_config.NumberColumn("Distance (km)", step=0.1, format="%.2f"),
            "rpe": st.column_config.NumberColumn("RPE", min_value=1, max_value=10, step=1),
            "Duration": st.column_config.TextColumn("Duration (HH:MM:SS)"),
            "Pace": st.column_config.Column("Pace (auto)", disabled=True),
            "session_type": st.column_config.TextColumn("Type"),
            "sport": st.column_config.SelectboxColumn("Sport", options=["Run", "Bike", "Swim", "S&C", "Other"]),
            "notes": st.column_config.TextColumn("Notes"),
        },
        use_container_width=True,
        key="editor",
    )

    # Save edits
    if st.button("Save Edits", type="primary"):
        # Merge edited values back into canonical columns and update DB
        # Map 'Duration' back to 'duration_s'
        merged = edited.merge(filtered[["id"] + [c for c in filtered.columns if c not in edited.columns]], on="id", how="left")
        # Rebuild duration_s and date format
        for _, row in merged.iterrows():
            # Convert Duration string to seconds
            dur_s = parse_duration_to_seconds(row.get("Duration", ""))
            # Build a row dict compatible with update_entry
            upd = {
                "id": row["id"],
                "session_date": row["session_date"],
                "sport": row["sport"],
                "session_type": row.get("session_type"),
                "distance_km": row.get("distance_km"),
                "duration_s": dur_s,
                "rpe": row.get("rpe"),
                "notes": row.get("notes"),
            }
            update_entry(conn, upd)
        st.success("Edits saved ✔️")
        st.rerun()

    # Delete selected rows
    with st.expander("Delete sessions…"):
        ids_to_delete = st.multiselect("Select entry IDs to delete", options=list(edited["id"]))
        if st.button("Delete Selected", type="secondary", disabled=len(ids_to_delete) == 0):
            delete_entries(conn, ids_to_delete)
            st.warning(f"Deleted {len(ids_to_delete)} session(s).")
            st.rerun()

# -----------------------------
# Import / Export
# -----------------------------
st.subheader("Import / Export")

c1, c2 = st.columns(2)
with c1:
    if st.button("Export CSV"):
        all_df = load_entries(conn)
        if not all_df.empty:
            # Convert duration to HH:MM:SS for a human-friendly CSV
            out = all_df.copy()
            out["Duration"] = out["duration_s"].apply(lambda x: format_seconds(int(x)) if pd.notna(x) else "")
            out["session_date"] = out["session_date"].astype(str)
            cols = ["id", "session_date", "sport", "session_type", "distance_km", "Duration", "rpe", "notes"]
            st.download_button("Download training_diary.csv", out[cols].to_csv(index=False), file_name="training_diary.csv", mime="text/csv")
        else:
            st.info("Nothing to export yet.")

with c2:
    up = st.file_uploader("Import CSV (same columns as export)", type=["csv"])
    if up is not None:
        try:
            imp = pd.read_csv(up)
            # Try best-effort mapping
            # Accept either Duration (HH:MM:SS) or duration_s
            for _, row in imp.iterrows():
                session_date = str(row.get("session_date") or row.get("date") or date.today())
                sport = row.get("sport", "Run")
                session_type = row.get("session_type", None)
                distance_km = row.get("distance_km", None)
                if pd.isna(distance_km): distance_km = None
                duration_s = row.get("duration_s", None)
                if pd.isna(duration_s) or duration_s is None:
                    duration_s = parse_duration_to_seconds(row.get("Duration", ""))
                rpe = row.get("rpe", None)
                if pd.isna(rpe): rpe = None
                notes = row.get("notes", None)

                insert_entry(conn, session_date, sport, session_type, distance_km, int(duration_s) if duration_s else None, int(rpe) if rpe else None, notes)
            st.success("Import complete ✔️")
            st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")
