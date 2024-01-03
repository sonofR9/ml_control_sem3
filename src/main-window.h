#pragma once

#include "allocator.h"
#include "options.h"
#include "tensor.h"

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>

// TODO(novak) QCharts
#include <QtCharts/QChartView>

#include <future>
#include <memory>
#include <thread>

class MainWindow : public QMainWindow {
  template <typename T, class Alloc = std::allocator<T>>
  using Tensor = optimization::Tensor<T, Alloc>;

  template <typename T>
  using RepetitiveAllocator = optimization::RepetitiveAllocator<T>;

 public:
  MainWindow(optimization::GlobalOptions& options, QWidget* parent = nullptr);

  void emitIterationChanged(std::size_t iteration);

 signals:
  void iterationChanged(std::size_t iteration);

 private:
  void constructView();
  void init();

 private slots:
  void onIterationChanged(std::size_t iteration);

 private:
  optimization::GlobalOptions& options_;

  std::future<Tensor<double, RepetitiveAllocator<double>>> optimResult_;
  std::vector<std::vector<double, RepetitiveAllocator<double>>> trajectory_;

  QLineEdit* tMax_;
  QLineEdit* dt_;
  QComboBox* method_;
  QLineEdit* printStep_;
  QPushButton* saveFile_;
  QLineEdit* filePath_;
  QCheckBox* clear_;

  struct Control {
    QLineEdit* number_;
    QLineEdit* min_;
    QLineEdit* max_;
  } control_;

  struct Wolf {
    QLineEdit* num_;
    QLineEdit* best_;
    QVBoxLayout* layout_;
  } wolf_;

  struct Evolution {
    QLineEdit* population_;
    QLineEdit* mutation_;
    QLineEdit* crossover_;
    QVBoxLayout* layout_;
  } evolution_;

  QProgressBar* progress_;
  QLabel* iterations_;
  QLabel* iterTime_;
};
