using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Microsoft.Win32;

namespace NeuroVision.Desktop {
    public partial class MainWindow : Window {
        // Ścieżka do wgranego obrazu
        private string selectedImagePath = null;

        public MainWindow() {
            InitializeComponent();
            LoadModelList();
        }

        // Ładuje listę modeli ONNX z folderu "Models"
        private void LoadModelList() {
            string modelsPath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\NeuroVision.Training\Models"));

            if (!Directory.Exists(modelsPath)) {
                MessageBox.Show("Folder 'Models' nie istnieje.");
                return;
            }

            // Szukamy plików .onnx
            var modelFiles = Directory.GetFiles(modelsPath, "*.onnx");
            cbModels.ItemsSource = modelFiles.Select(Path.GetFileName);
            if (modelFiles.Any())
                cbModels.SelectedIndex = 0;
        }

        // Obsługa kliknięcia przycisku "Wgraj zdjęcie"
        private void btnLoadImage_Click(object sender, RoutedEventArgs e) {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp";
            if (dlg.ShowDialog() == true) {
                selectedImagePath = dlg.FileName;
                // Ustawienie podglądu obrazu
                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(selectedImagePath);
                bitmap.DecodePixelWidth = 150;
                bitmap.EndInit();
                imgPreview.Source = bitmap;
            }
        }

        // Przetwarzanie obrazu: resize do 32x32 i normalizacja pikseli do [-1, 1]
        private float[] ProcessImage(string imagePath) {
            using (Bitmap bitmap = new Bitmap(imagePath)) {
                // Zmiana rozmiaru na 32x32
                Bitmap resized = new Bitmap(32, 32);
                using (Graphics g = Graphics.FromImage(resized)) {
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.DrawImage(bitmap, 0, 0, 32, 32);
                }

                // Tworzymy tablicę dla tensora o rozmiarze [3, 32, 32]
                float[] imgData = new float[3 * 32 * 32];
                for (int y = 0; y < 32; y++) {
                    for (int x = 0; x < 32; x++) {
                        System.Drawing.Color color = resized.GetPixel(x, y);
                        // Normalizacja: [0,255] -> [0,1] -> [-1,1]
                        float r = (color.R / 255f - 0.5f) / 0.5f;
                        float gVal = (color.G / 255f - 0.5f) / 0.5f;
                        float b = (color.B / 255f - 0.5f) / 0.5f;
                        int idx = y * 32 + x;
                        imgData[idx] = r;
                        imgData[32 * 32 + idx] = gVal;
                        imgData[2 * 32 * 32 + idx] = b;
                    }
                }
                return imgData;
            }
        }

        // Obsługa przycisku "Predykcja"
        private void btnPredict_Click(object sender, RoutedEventArgs e) {
            if (selectedImagePath == null) {
                MessageBox.Show("Najpierw wgraj zdjęcie.");
                return;
            }

            if (cbModels.SelectedItem == null) {
                MessageBox.Show("Wybierz model z listy.");
                return;
            }

            // Ustalamy pełną ścieżkę do modelu
            string modelsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
            string modelFileName = cbModels.SelectedItem.ToString();
            string modelPath = Path.Combine(modelsPath, modelFileName);

            // Przetwarzamy obraz i tworzymy tensor o wymiarach [1, 3, 32, 32]
            float[] imgData = ProcessImage(selectedImagePath);
            var inputTensor = new DenseTensor<float>(imgData, new int[] { 1, 3, 32, 32 });
            var inputs = new List<NamedOnnxValue>
            {
                // Nazwa wejścia "input" – należy dostosować do nazwy wejścia Twojego modelu
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // Wykonywanie inferencji przy użyciu ONNX Runtime
            using (var session = new InferenceSession(modelPath)) {
                var results = session.Run(inputs);
                // Zakładamy, że model zwraca tensor o kształcie [1, 10] (logity dla 10 klas)
                var outputTensor = results.First().AsEnumerable<float>().ToArray();
                int predictedClass = Array.IndexOf(outputTensor, outputTensor.Max());
                tbResult.Text = $"Wynik: Klasa {predictedClass}";
            }
        }
    }
}