import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st

page = st.sidebar.selectbox("Pilih Halaman", ["Pendahuluan", "Dashboard"])

if page == "Pendahuluan":

    st.image('https://github.com/sanfla/Customer_Segemtation/blob/main/customer.png?raw=true', use_column_width=True)

    st.title("Pendahuluan")

    data1 = pd.read_csv("https://raw.githubusercontent.com/sanfla/Customer_Segemtation/main/marketing_campaign.csv", delimiter='\t')
    st.dataframe(data1.head(5))

    st.write("""
    **Customer Personality Analysis** adalah analisis mendetail tentang pelanggan ideal perusahaan. Ini membantu bisnis untuk lebih memahami pelanggan mereka dan memudahkan mereka untuk memodifikasi produk sesuai dengan kebutuhan, perilaku, dan kekhawatiran spesifik dari berbagai jenis pelanggan.

    Analisis kepribadian pelanggan membantu bisnis untuk memodifikasi produk berdasarkan pelanggan target dari berbagai segmen pelanggan. Misalnya, daripada mengeluarkan uang untuk memasarkan produk baru kepada setiap pelanggan dalam database perusahaan, perusahaan dapat menganalisis segmen pelanggan mana yang paling mungkin membeli produk tersebut dan kemudian memasarkan produk hanya kepada segmen tersebut.

    ### Fitur

    **People**
    - **ID**: Identifikasi unik pelanggan
    - **Year_Birth**: Tahun kelahiran pelanggan
    - **Education**: Tingkat pendidikan pelanggan
    - **Marital_Status**: Status pernikahan pelanggan
    - **Income**: Pendapatan tahunan rumah tangga pelanggan
    - **Kidhome**: Jumlah anak dalam rumah tangga pelanggan
    - **Teenhome**: Jumlah remaja dalam rumah tangga pelanggan
    - **Dt_Customer**: Tanggal pendaftaran pelanggan dengan perusahaan
    - **Recency**: Jumlah hari sejak pembelian terakhir pelanggan
    - **Complain**: 1 jika pelanggan mengeluh dalam 2 tahun terakhir, 0 jika tidak

    **Products**
    - **MntWines**: Jumlah yang dibelanjakan untuk anggur dalam 2 tahun terakhir
    - **MntFruits**: Jumlah yang dibelanjakan untuk buah dalam 2 tahun terakhir
    - **MntMeatProducts**: Jumlah yang dibelanjakan untuk produk daging dalam 2 tahun terakhir
    - **MntFishProducts**: Jumlah yang dibelanjakan untuk produk ikan dalam 2 tahun terakhir
    - **MntSweetProducts**: Jumlah yang dibelanjakan untuk produk manis dalam 2 tahun terakhir
    - **MntGoldProds**: Jumlah yang dibelanjakan untuk emas dalam 2 tahun terakhir

    **Promotion**
    - **NumDealsPurchases**: Jumlah pembelian yang dilakukan dengan diskon
    - **AcceptedCmp1**: 1 jika pelanggan menerima tawaran dalam kampanye ke-1, 0 jika tidak
    - **AcceptedCmp2**: 1 jika pelanggan menerima tawaran dalam kampanye ke-2, 0 jika tidak
    - **AcceptedCmp3**: 1 jika pelanggan menerima tawaran dalam kampanye ke-3, 0 jika tidak
    - **AcceptedCmp4**: 1 jika pelanggan menerima tawaran dalam kampanye ke-4, 0 jika tidak
    - **AcceptedCmp5**: 1 jika pelanggan menerima tawaran dalam kampanye ke-5, 0 jika tidak
    - **Response**: 1 jika pelanggan menerima tawaran dalam kampanye terakhir, 0 jika tidak
             
    Data diatas telah diubah menjadi bentuk yang lebih sederhana
    """)
    data2 = pd.read_csv("https://raw.githubusercontent.com/sanfla/Customer_Segemtation/main/data_new.csv")
    st.dataframe(data2.head(5))



elif page == "Dashboard":

    st.image('https://github.com/sanfla/Customer_Segemtation/blob/main/customer.png?raw=true', use_column_width=True)

    st.title("Customer Segment Dashboard")

    def hapus_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def elbow(X, max_cls):
        wcss = []
        for k in range(1, max_cls + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            wcss.append(kmeans.inertia_)
        return wcss

    def main():
        # Load and preprocess data
        data2 = pd.read_csv("https://raw.githubusercontent.com/sanfla/Customer_Segemtation/main/data_new.csv")

        binary = [col for col in data2.columns if data2[col].nunique() == 2]
        categorical = [col for col in data2.columns if 2 < data2[col].nunique() < 10]
        numerical = [col for col in data2.select_dtypes(include=['number']).columns 
                    if col not in binary + categorical]

        data2 = hapus_outliers(data2, numerical)

        categorical = data2.select_dtypes(include=['object']).columns
        encoded_data = pd.get_dummies(data2, columns=categorical, drop_first=True, dtype=int)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(encoded_data)

        # Elbow method to determine optimal number of clusters
        max_clusters = 10  # Use a fixed value for non-interactive version
        wcss = elbow(scaled_data, max_clusters)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_clusters + 1), wcss, marker='o')
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig("elbow_method.png")
        st.image("elbow_method.png")

        num_clusters = 3  

        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        
        st.write(f"Silhouette Score for {num_clusters} clusters: {silhouette_avg:.2f}")

        data2['Cluster'] = cluster_labels

        transform_data = scaler.inverse_transform(scaled_data)
        
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("hsv", num_clusters)
        for i in range(num_clusters):
            plt.scatter(transform_data[data2['Cluster'] == i, encoded_data.columns.get_loc('Income')],
                        transform_data[data2['Cluster'] == i, encoded_data.columns.get_loc('Pengeluaran_2thn')],
                        s=25, c=colors[i], label=f'Cluster {i+1}')
        plt.title('Clusters of Customers')
        plt.xlabel('Income')
        plt.ylabel('Pengeluaran_2thn')
        plt.legend()
        plt.savefig("clusters.png")
        st.image("clusters.png")

        plt.figure(figsize=(10, 6))
        for i in range(num_clusters):
            plt.scatter(transform_data[data2['Cluster'] == i, encoded_data.columns.get_loc('umur')],
                        transform_data[data2['Cluster'] == i, encoded_data.columns.get_loc('Recency')],
                        s=25, c=colors[i], label=f'Cluster {i+1}')
        plt.title('Clusters of Customers (Age vs Recency)')
        plt.xlabel('Age')
        plt.ylabel('Recency')
        plt.legend()
        plt.savefig("clusters_age_recency.png")
        st.image("clusters_age_recency.png")

    if __name__ == "__main__":
        main()