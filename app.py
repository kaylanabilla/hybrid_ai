# =========================
# ATURAN LOGIKA MANUSIA (POST-PROCESS)
# =========================
final_label = confidence >= 60  # hasil model asli

# Ambil nilai input penting
if "Sleep Duration (hrs)" in columns:
    sleep = user_input[columns.index("Sleep Duration (hrs)")]
else:
    sleep = None

if "Exercise (mins)" in columns:
    exercise = user_input[columns.index("Exercise (mins)")]
else:
    exercise = None

# Override logika ekstrem
if sleep is not None and exercise is not None:
    if sleep >= 7 and exercise < 10:
        final_label = False  # pasti tidak produktif
    elif sleep < 5 and exercise >= 40:
        final_label = True   # masih bisa produktif

# =========================
# OUTPUT FINAL
# =========================
if final_label:
    st.success("✅ PRODUKTIF (sesuai kondisi input)")
else:
    st.error("❌ TIDAK PRODUKTIF (sesuai kondisi input)")
