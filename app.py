import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tra cứu & so sánh giá nhà", page_icon="🏠", layout="wide")

CURRENCY = "$"            # đổi ký hiệu tiền tệ ở đây nếu cần
DATA_FILE = "gia_nha.csv"  # file dữ liệu đặt cùng thư mục với app.py

# (tên cột, nhãn hiển thị, kiểu nhập, bước nhảy)
FEATURES = [
    ("OverallQual",       "Chất lượng tổng thể (1-10)",            "int",   1),
    ("TotalSF",           "Tổng diện tích sử dụng",                 "int",   50),
    ("GrLivArea",         "Diện tích sinh hoạt trên mặt đất",       "int",   50),
    ("LotArea",           "Diện tích lô đất",                       "int",   1000),
    ("HouseAge",          "Tuổi nhà (năm)",                         "int",   1),
    ("BedroomAbvGr",      "Số phòng ngủ",                           "int",   1),
    ("TotalBaths",        "Số phòng tắm",                           "float", 0.5),
    ("GarageCars",        "Số chỗ để xe trong gara",                "int",   1),
    ("Fireplaces",        "Số lò sưởi",                             "int",   1),
    ("KitchenQual_Score", "Chất lượng bếp (điểm)",                  "int",   1),
]
KEYS = [f[0] for f in FEATURES]
LABEL = {f[0]: f[1] for f in FEATURES}


def fmt(v):
    return f"{CURRENCY}{v:,.0f}"


# =============== THUẬT TOÁN HỒI QUY TUYẾN TÍNH + GRADIENT DESCENT ===============
def init_weights(n_features, seed=42, scale=0.01):
    """Bước 2: khởi tạo trọng số w ngẫu nhiên nhỏ, bias b = 0."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, n_features), 0.0


def predict(X, w, b):
    """Bước 1 + 3: y_hat = w1*x1 + ... + wn*xn + b."""
    return X @ w + b


def mse_loss(y, y_hat):
    """Hàm mất mát: L = (1/2m) * sum((y_hat - y)^2)."""
    return float(np.mean((y_hat - y) ** 2) / 2)


def gradients(X, y, y_hat):
    m = len(y)
    err = y_hat - y
    return X.T @ err / m, float(err.mean())


def gradient_descent(X, y, lr=0.05, epochs=1000, seed=42):
    """Bước 4: w := w - lr*dL/dw ; b := b - lr*dL/db."""
    w, b = init_weights(X.shape[1], seed)
    loss_hist = []
    for _ in range(epochs):
        y_hat = predict(X, w, b)
        loss_hist.append(mse_loss(y, y_hat))
        dw, db = gradients(X, y, y_hat)
        w = w - lr * dw
        b = b - lr * db
    loss_hist.append(mse_loss(y, predict(X, w, b)))
    return w, b, loss_hist


def r2_score(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# =============== DỮ LIỆU ===============
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def get_data():
    if os.path.exists(DATA_FILE):
        return load_csv(DATA_FILE)
    up = st.sidebar.file_uploader("Tải file Nh_m4_GiaNha_Train.xlsx", type=["xlsx"])
    if up is None:
        st.title("🏠 Tra cứu & so sánh giá nhà")
        st.info("Chưa thấy file dữ liệu. Hãy tải file Excel ở thanh bên trái.")
        st.stop()
    raw = pd.read_excel(up)
    d = raw[KEYS + ["SalePrice"]].copy()
    return d.fillna(d.median())


@st.cache_data
def train(df, lr=0.05, epochs=1000, seed=42):
    X = df[KEYS].values.astype(float)
    y = df["SalePrice"].values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1
    ym, ys = y[tr].mean(), y[tr].std()
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr = (y[tr] - ym) / ys

    w0, b0 = init_weights(len(KEYS), seed)
    loss0 = mse_loss(ytr, predict(Xtr, w0, b0))
    w, b, loss_hist = gradient_descent(Xtr, ytr, lr, epochs, seed)

    p_tr = predict(Xtr, w, b) * ys + ym
    p_te = predict(Xte, w, b) * ys + ym
    return dict(
        mu=mu, sd=sd, ym=ym, ys=ys, w=w, b=b, w0=w0, b0=b0, loss0=loss0,
        loss_hist=loss_hist, lr=lr, epochs=epochs,
        r2_tr=r2_score(y[tr], p_tr), r2_te=r2_score(y[te], p_te),
        rmse_te=float(np.sqrt(np.mean((y[te] - p_te) ** 2))),
        rmse_base=float(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2))),
        n_tr=len(tr), n_te=len(te),
    )


df = get_data()
M = train(df)
prices = df["SalePrice"].values
Zall = (df[KEYS].values.astype(float) - M["mu"]) / M["sd"]

# =============== GIAO DIỆN ===============
st.title("🏠 Tra cứu & so sánh giá nhà")
st.caption("Nhập thông tin căn nhà, ứng dụng dự đoán giá bằng mô hình hồi quy tuyến tính "
           "(huấn luyện bằng Gradient Descent trên dữ liệu 800 căn nhà) rồi so sánh với giá bạn mong muốn.")

st.sidebar.header("Tùy chọn so sánh")
role = st.sidebar.radio("Bạn đang muốn", ["Mua nhà", "Bán nhà"], horizontal=True)
tol = st.sidebar.slider("Ngưỡng coi là 'gần nhau' (±%)", 1, 30, 10)
st.sidebar.markdown("---")
st.sidebar.caption("Đơn vị diện tích và giá theo đúng đơn vị trong bộ dữ liệu.")

tab_main, tab_sim, tab_model = st.tabs(["🔎 Tra cứu & so sánh", "🏘️ Nhà tương tự trong dữ liệu", "📐 Về mô hình"])

# ---------- Nhập thông tin ----------
with tab_main:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.subheader("Thông tin căn nhà")
        vals = {}
        for key, label, kind, step in FEATURES:
            lo, hi, med = float(df[key].min()), float(df[key].max()), float(df[key].median())
            if kind == "int":
                lo, hi, med = int(lo), int(hi), int(round(med))
                step = int(step)
            else:
                step = float(step)
            if key in ("OverallQual", "KitchenQual_Score"):
                vals[key] = st.slider(label, lo, hi, med, key=key)
            else:
                vals[key] = st.number_input(label, min_value=lo, max_value=hi, value=med, step=step, key=key)
        st.subheader("Giá bạn mong muốn")
        desired = st.number_input(f"Giá mong muốn ({CURRENCY})", min_value=0, max_value=5_000_000,
                                  value=int(round(float(np.median(prices)), -3)), step=5000, key="desired")

    # ---------- Dự đoán ----------
    x = np.array([vals[k] for k in KEYS], dtype=float)
    z = (x - M["mu"]) / M["sd"]
    pred = float(predict(z, M["w"], M["b"]) * M["ys"] + M["ym"])
    pred = float(np.clip(pred, prices.min(), prices.max()))
    lo_r = max(pred - M["rmse_te"], float(prices.min()))
    hi_r = min(pred + M["rmse_te"], float(prices.max()))
    diff = desired - pred
    pct = diff / pred * 100 if pred else 0.0
    pct_pred = float((prices <= pred).mean() * 100)
    pct_des = float((prices <= desired).mean() * 100)

    with right:
        st.subheader("Kết quả")
        c1, c2, c3 = st.columns(3)
        c1.metric("Giá dự đoán", fmt(pred))
        c2.metric("Giá mong muốn", fmt(desired))
        c3.metric("Chênh lệch", fmt(abs(diff)), f"{pct:+.1f}%", delta_color="off")
        st.caption(f"Khoảng tham khảo quanh giá dự đoán (± sai số RMSE trên tập kiểm tra): {fmt(lo_r)} đến {fmt(hi_r)}")

        if abs(pct) <= tol:
            st.success(f"✅ Giá mong muốn **gần với** giá mô hình dự đoán (lệch {pct:+.1f}%, trong ngưỡng ±{tol}%).")
        elif desired < pred:
            if role == "Mua nhà":
                st.info(f"💡 Giá bạn mong muốn **thấp hơn** giá dự đoán {abs(pct):.1f}%. Có lợi cho người mua, "
                        "nhưng bên bán có thể không chấp nhận.")
            else:
                st.info(f"💡 Giá bạn muốn bán **thấp hơn** giá dự đoán {abs(pct):.1f}%. Bạn có thể nâng giá lên.")
        else:
            if role == "Mua nhà":
                st.warning(f"⚠️ Giá bạn mong muốn **cao hơn** giá dự đoán {pct:.1f}%. Bạn có thể đang trả cao hơn mức mô hình ước tính.")
            else:
                st.warning(f"⚠️ Giá bạn muốn bán **cao hơn** giá dự đoán {pct:.1f}%. Có thể khó bán nếu so với mặt bằng dữ liệu.")

        st.write(f"Trong 800 căn của bộ dữ liệu: giá dự đoán nằm ở **phân vị {pct_pred:.0f}%**, "
                 f"giá mong muốn nằm ở **phân vị {pct_des:.0f}%** (tức {pct_des:.0f}% số căn có giá thấp hơn hoặc bằng).")

        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.hist(prices, bins=25, color="#c9d6ea", edgecolor="white")
        ax.axvline(pred, color="#1f5fa8", lw=2.5, label=f"Dự đoán {fmt(pred)}")
        ax.axvline(desired, color="#d9442b", lw=2.5, ls="--", label=f"Mong muốn {fmt(desired)}")
        ax.set_xlabel(f"Giá bán ({CURRENCY})"); ax.set_ylabel("Số căn nhà")
        ax.set_title("Vị trí so với phân bố giá trong dữ liệu"); ax.legend(); ax.grid(alpha=.25)
        st.pyplot(fig)

        if M["r2_te"] < 0.3:
            st.warning(
                f"**Độ tin cậy của mô hình thấp.** R² trên tập kiểm tra = {M['r2_te']:.2f} "
                f"(1 là tốt nhất, ≤ 0 là không hơn việc đoán bằng giá trung bình). "
                f"Sai số RMSE = {fmt(M['rmse_te'])}, trong khi đoán bằng giá trung bình cho sai số {fmt(M['rmse_base'])}. "
                "Các đặc điểm nhà trong bộ dữ liệu này gần như không giải thích được giá, "
                "vì vậy hãy xem kết quả như một con số tham khảo, không phải định giá chính xác.")

# ---------- Nhà tương tự ----------
with tab_sim:
    st.subheader("5 căn nhà giống căn bạn nhập nhất trong dữ liệu")
    st.caption("Tìm theo khoảng cách (Euclid) trên các đặc điểm đã chuẩn hóa. Đây là giá bán thực tế trong dữ liệu.")
    dist = np.sqrt(((Zall - z) ** 2).sum(axis=1))
    nn = np.argsort(dist)[:5]
    sim = df.iloc[nn][KEYS + ["SalePrice"]].copy()
    sim.insert(0, "Độ khác biệt", np.round(dist[nn], 2))
    sim = sim.rename(columns={**LABEL, "SalePrice": "Giá bán thực tế"})
    st.dataframe(sim.reset_index(drop=True), hide_index=True)
    mean_sim = float(df.iloc[nn]["SalePrice"].mean())
    d2 = desired - mean_sim
    st.write(f"Giá trung bình của 5 căn tương tự: **{fmt(mean_sim)}**. "
             f"Giá bạn mong muốn {'cao hơn' if d2 > 0 else 'thấp hơn'} **{fmt(abs(d2))}** "
             f"({d2 / mean_sim * 100:+.1f}%).")

# ---------- Về mô hình ----------
with tab_model:
    st.subheader("1. Phương trình hồi quy tuyến tính đa biến")
    st.latex(r"\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b")
    st.write(f"Biến mục tiêu y là giá bán; {len(KEYS)} biến đầu vào x gồm: " + ", ".join(LABEL[k] for k in KEYS) +
             ". Dữ liệu được chuẩn hóa z-score, chia 80% huấn luyện / 20% kiểm tra.")

    st.subheader("2. Khởi tạo trọng số")
    st.write("w lấy ngẫu nhiên nhỏ (độ lệch 0,01), b = 0.")
    st.dataframe(pd.DataFrame({"Biến": [LABEL[k] for k in KEYS], "w khởi tạo": M["w0"]}).round(5),
                 hide_index=True)

    st.subheader("3. Dự đoán và hàm mất mát")
    st.latex(r"L(\mathbf{w},b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2")
    st.write(f"Mất mát ban đầu với trọng số khởi tạo: **{M['loss0']:.4f}** → sau huấn luyện: **{M['loss_hist'][-1]:.4f}**.")

    st.subheader("4. Cập nhật trọng số bằng Gradient Descent")
    st.latex(r"\mathbf{w} := \mathbf{w}-\alpha\,\frac{1}{m}\mathbf{X}^T(\hat{\mathbf{y}}-\mathbf{y}),\qquad "
             r"b := b-\alpha\,\frac{1}{m}\sum(\hat{y}_i-y_i)")
    st.write(f"Learning rate α = {M['lr']}, số vòng lặp = {M['epochs']}.")
    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    ax2.plot(M["loss_hist"]); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.grid(alpha=.3)
    ax2.set_title("Đường giảm hàm mất mát")
    st.pyplot(fig2)

    st.subheader("Trọng số sau huấn luyện (thang đo chuẩn hóa)")
    wt = pd.DataFrame({"Biến": [LABEL[k] for k in KEYS], "w": M["w"]})
    st.dataframe(wt.reindex(wt["w"].abs().sort_values(ascending=False).index).round(4),
                 hide_index=True)
    st.write(f"b = {M['b']:.4f}")

    st.subheader("Đánh giá")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("R² huấn luyện", f"{M['r2_tr']:.3f}")
    e2.metric("R² kiểm tra", f"{M['r2_te']:.3f}")
    e3.metric("RMSE kiểm tra", fmt(M["rmse_te"]))
    e4.metric("RMSE nếu đoán trung bình", fmt(M["rmse_base"]))
