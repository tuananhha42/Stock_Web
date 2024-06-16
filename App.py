import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import date
from sklearn.model_selection import train_test_split

st.header("Thông tin giao dịch chứng khoán")
# Tạo ứng dụng Streamlit
sidebar_expander = st.sidebar.expander("Nhập dữ liệu chứng khoán")

# Đọc dữ liệu từ tệp Excel
data = pd.read_excel("merged_data.xlsx")
company_codes = dict(zip(data["Name"], data["Ticket"]))

# Danh sách các công ty
companies = list(company_codes.keys())
# Tạo select box cho việc chọn công ty
selected_company = sidebar_expander.selectbox("Chọn một công ty:", companies)
# Lấy mã công ty tương ứng với công ty đã chọn
selected_company_code = company_codes[selected_company]

# Chọn ngày bắt đầu và kết thúc
selected_date_start = sidebar_expander.date_input("Chọn ngày bắt đầu", date.today(), key='selected_date_start')
selected_date_end = sidebar_expander.date_input("Chọn ngày kết thúc", date.today(), key='selected_date_end')

if sidebar_expander.button('Xem thông tin dữ liệu'):
    # Sử dụng Yahoo Finance để lấy dữ liệu chứng khoán
    try:
        st.session_state.df = yf.download(selected_company_code, start=selected_date_start, end=selected_date_end)
        
        # Hiển thị dữ liệu chứng khoán
        st.write(f'Dữ liệu chứng khoán của công ty {selected_company} từ {selected_date_start} đến {selected_date_end}:')
        st.write(st.session_state.df)

        # Vẽ biểu đồ giá đóng cửa hàng ngày
        st.write('Biểu đồ giá đóng cửa hàng ngày:')
        fig = go.Figure(data=go.Candlestick(x=st.session_state.df.index, open=st.session_state.df['Open'], high=st.session_state.df['High'], low=st.session_state.df['Low'], close=st.session_state.df['Close']))
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Không thể lấy dữ liệu chứng khoán cho {selected_company_code}: {e}")
else:
    # Tải dữ liệu từ yfinance
    today = date.today()
    start_date = today.replace(year=today.year - 10)  # Lấy dữ liệu từ năm trước đến hiện tại
    end_date = today
    try:
        st.session_state.df = yf.download(selected_company_code, start=start_date, end=end_date)
        st.write(f'Dữ liệu chứng khoán của công ty {selected_company} trong 10 năm qua:')
        st.write(st.session_state.df)
    except Exception as e:
        st.error(f"Không thể tải dữ liệu chứng khoán cho {selected_company_code}: {e}")

# Phân tích dữ liệu trong expander
analysis_expander = st.sidebar.expander("Phân tích dữ liệu")

if analysis_expander:
    # Chọn các cột dữ liệu số
    numeric_columns = st.session_state.df.select_dtypes(include=['float64', 'int64']).columns
    selected_numeric_columns = analysis_expander.multiselect("Chọn các cột dữ liệu số:", numeric_columns)

    # Chọn tùy chọn phân tích
    analysis_options = ["shape", "info", "describe", "min", "max", "mean", "median", "mode", "count null", "q1", "q2", "q3", "iqr", "var", "std"]
    selected_analysis_option = analysis_expander.selectbox("Chọn tùy chọn phân tích:", analysis_options)
    if selected_numeric_columns:
        if selected_analysis_option == "shape":
            st.write("Kích thước của DataFrame:")
            st.write(st.session_state.df.shape)
        elif selected_analysis_option == "info":
            st.write("Thông tin DataFrame:")
            st.write(st.session_state.df.info())
        elif selected_analysis_option == "describe":
            st.write("Mô tả tổng quan của DataFrame:")
            st.write(st.session_state.df.describe())
        elif selected_analysis_option == "min":
            st.write("Giá trị nhỏ nhất của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].min())
        elif selected_analysis_option == "max":
            st.write("Giá trị lớn nhất của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].max())
        elif selected_analysis_option == "mean":
            st.write("Giá trị trung bình của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].mean())
        elif selected_analysis_option == "median":
            st.write("Median của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].median())
        elif selected_analysis_option == "mode":
            st.write("Mode của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].mode())
        elif selected_analysis_option == "count null":
            st.write("Số lượng giá trị thiếu của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].isnull().sum())
        elif selected_analysis_option in ["q1", "q2", "q3"]:
            quantile_num = int(selected_analysis_option[1])
            st.write(f"Phân vị {quantile_num} của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].quantile(quantile_num/4))
        elif selected_analysis_option == "iqr":
            st.write("Phạm vi tương quan giữa Q3 và Q1 của các cột số:")
            iqr = st.session_state.df[selected_numeric_columns].quantile(0.75) - st.session_state.df[selected_numeric_columns].quantile(0.25)
            st.write(iqr)
        elif selected_analysis_option == "var":
            st.write("Phương sai của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].var())
        elif selected_analysis_option == "std":
            st.write("Độ lệch chuẩn của các cột số:")
            st.write(st.session_state.df[selected_numeric_columns].std())
    
# Tiền xử lý dữ liệu trong expander
preprocessing_expander = st.sidebar.expander("Tiền xử lý dữ liệu")

if preprocessing_expander:
    # Chọn biến độc lập X
    independent_variables = st.session_state.df.columns.tolist()
    selected_independent_variables = preprocessing_expander.multiselect("Chọn biến độc lập X:", independent_variables)

    # Chọn biến phụ thuộc y
    dependent_variables = list(set(independent_variables) - set(selected_independent_variables))
    selected_dependent_variable = preprocessing_expander.selectbox("Chọn biến phụ thuộc y:", dependent_variables)

    st.session_state.X = st.session_state.df[selected_independent_variables]
    st.session_state.y = st.session_state.df[selected_dependent_variable]

    # Nút chia dữ liệu train-test
    if preprocessing_expander.button("Chia dữ liệu train-test (80-20)"):
        try:
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=0.2, random_state=42)

            st.success("Dữ liệu đã được chia thành train và test!")
        #     if 'X_train' and 'X_test' and 'y_train' and 'y_test' in st.session_state:
        # # Hiển thị X_train và X_test
        #         st.subheader("Dữ liệu X_train và X_test:")
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             st.write("X_train:")
        #             st.write(st.session_state.X_train)
        #         with col2:
        #             st.write("X_test:")
        #             st.write(st.session_state.X_test)

        #         # Hiển thị y_train và y_test
        #         st.subheader("Dữ liệu y_train và y_test:")
        #         col3, col4 = st.columns(2)
        #         with col3:
        #             st.write("y_train:")
        #             st.write(st.session_state.y_train)
        #         with col4:
        #             st.write("y_test:")
        #             st.write(st.session_state.y_test)
        except Exception as e:
            st.error(f"Lỗi khi chia dữ liệu train-test: {e}")
    if 'X_train' and 'X_test' and 'y_train' and 'y_test' in st.session_state:
    # Hiển thị X_train và X_test
        st.subheader("Dữ liệu X_train và X_test:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("X_train:")
            st.write(st.session_state.X_train)
        with col2:
            st.write("X_test:")
            st.write(st.session_state.X_test)

        # Hiển thị y_train và y_test
        st.subheader("Dữ liệu y_train và y_test:")
        col3, col4 = st.columns(2)
        with col3:
            st.write("y_train:")
            st.write(st.session_state.y_train)
        with col4:
            st.write("y_test:")
            st.write(st.session_state.y_test)

    # Nút chuẩn hóa dữ liệu
    if preprocessing_expander.button("Chuẩn hóa dữ liệu"):
        try:
            from sklearn.preprocessing import MinMaxScaler

            st.session_state.scaler = MinMaxScaler()
            st.session_state.X_train_scaled = st.session_state.scaler.fit_transform(st.session_state.X_train)
            st.session_state.X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)

            st.success("Dữ liệu đã được chuẩn hóa!")
            # Hiển thị X_train_scaled và X_test_scaled
            

        except Exception as e:
            st.error(f"Lỗi khi chuẩn hóa dữ liệu: {e}")
    if 'X_train_scaled' and 'X_test_scaled'in st.session_state: 
        st.subheader("Dữ liệu X_train_scaled và X_test_scaled:")
        col5, col6 = st.columns(2)
        with col5:
            st.write("X_train sau khi chuẩn hóa:")
            st.write(st.session_state.X_train_scaled)
        with col6:
            st.write("X_test sau khi chuẩn hóa:")
            st.write(st.session_state.X_test_scaled)
        
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# Huấn luyện mô hình
def train_model(model_name, X_train, X_test, y_train, y_test):
    model = None
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge()
    elif model_name == "Lasso":
        model = Lasso()

    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred
    else:
        return None, None

# Tạo expander mới cho huấn luyện mô hình
model_training_expander = st.sidebar.expander("Huấn luyện mô hình")

if model_training_expander:
    # Chọn mô hình
    model_options = ["Linear Regression", "Ridge", "Lasso"]
    selected_model = model_training_expander.selectbox("Chọn mô hình:", model_options)

    # Chọn thông số đánh giá
    evaluation_metrics = ["R2", "MSE", "MAE", "RMSE"]
    selected_metric = model_training_expander.selectbox("Chọn thông số đánh giá:", evaluation_metrics)

    # Nút huấn luyện
    if model_training_expander.button("Huấn luyện"):
        try:
            st.session_state.model, st.session_state.y_pred = train_model(selected_model, st.session_state.X_train_scaled, st.session_state.X_test_scaled, st.session_state.y_train, st.session_state.y_test)
            
            if st.session_state.model and st.session_state.y_pred is not None:
                # Đánh giá mô hình
                if selected_metric == "R2":
                    st.session_state.evaluation_score = r2_score(st.session_state.y_test, st.session_state.y_pred)
                elif selected_metric == "MSE":
                    st.session_state.evaluation_score = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
                elif selected_metric == "MAE":
                    st.session_state.evaluation_score = mean_absolute_error(st.session_state.y_test, st.session_state.y_pred)
                elif selected_metric == "RMSE":
                    st.session_state.evaluation_score = sqrt(mean_squared_error(st.session_state.y_test, st.session_state.y_pred))

                # Hiển thị kết quả đánh giá
                
                # st.write(f"{selected_metric}: {st.session_state.evaluation_score}")
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình: {e}")

    if 'evaluation_score' in st.session_state:
        st.subheader("Kết quả đánh giá mô hình:")
        st.write(f"{selected_metric}: {st.session_state.evaluation_score}")

# Tạo expander mới cho dự đoán
prediction_expander = st.sidebar.expander("Dự đoán")
import numpy as np


if prediction_expander:
    if 'X_train' and 'X_test' and 'y_train' and 'y_test' in st.session_state:
        # Hiển thị ô nhập số cho các cột đã chọn của X
        st.session_state.input_values = []
        for column in st.session_state.X_train.columns:
            st.session_state.input_values.append(prediction_expander.number_input(f"Nhập giá trị cho {column}:", key=column))

        # Nút dự đoán
        if prediction_expander.button("Dự đoán"):
            try:
                # Tạo DataFrame mới từ giá trị đầu vào
                st.session_state.input_data = np.array(st.session_state.input_values).reshape(1, -1)
                
                st.session_state.input_data_scaled = st.session_state.scaler.transform(st.session_state.input_data)

                # Dự đoán thông qua mô hình đã chọn
                if st.session_state.model:
                    prediction = st.session_state.model.predict(st.session_state.input_data_scaled)
                    st.success(f"Dự đoán của mô hình: {prediction[0]}")
                else:
                    st.warning("Vui lòng huấn luyện một mô hình trước khi dự đoán.")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
    