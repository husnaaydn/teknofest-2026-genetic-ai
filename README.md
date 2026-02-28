# 🧬 TEKNOFEST 2026 – Genetik Varyant Sınıflandırma

## 📌 Proje Amacı

Bu proje, missense genetik varyantların:

- **Patojenik (Pathogenic)**
- **Benign (Zararsız)**

olarak sınıflandırılmasını amaçlayan bir makine öğrenmesi modelini içermektedir.

Proje, **TEKNOFEST 2026 – Sağlıkta Yapay Zeka** yarışması kapsamında geliştirilmiştir.

---

## 🧠 Kullanılan Yöntem

### Model
- Random Forest Classifier
- Stratified 5-Fold Cross Validation
- Fold bazlı Threshold (eşik) optimizasyonu
- Tüm veri ile final model eğitimi

### Değerlendirme Metrik
- **F1 Skoru** (yarışma ana metriği)

### Çapraz Doğrulama Performansı

- 5-Fold Ortalama CV F1 Skoru: **0.7979**