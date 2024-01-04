#pragma once

#include "allocator.h"
#include "options.h"
#include "tensor.h"

#include <QMainWindow>

// TODO(novak) QCharts
#include <QtCharts/QChartView>

#include <chrono>
#include <future>
#include <memory>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QHBoxLayout;
class QVBoxLayout;
class QWidget;

namespace optimization {
extern unsigned int seed;
}

constexpr const char* kAppFolder{"ml_control_sem3"};
constexpr const char* kConfigPathFile{"/config_file_path.ini"};

class MainWindow : public QMainWindow {
  Q_OBJECT

  template <typename T, class Alloc = std::allocator<T>>
  using Tensor = optimization::Tensor<T, Alloc>;

  template <typename T>
  using RepetitiveAllocator = optimization::RepetitiveAllocator<T>;

 public:
  explicit MainWindow(optimization::GlobalOptions& options,
                      QWidget* parent = nullptr);

  ~MainWindow() override;

  void emitIterationChanged(std::size_t iteration, double functional);

 signals:
  void iterationChanged(int iteration, double functional);

 private:
  void constructView();
  QWidget* constructOptimizationTab(QWidget* tabWidget);
  QVBoxLayout* constructGlobalParams(QWidget*);
  QVBoxLayout* constructControlParams(QWidget*);
  QWidget* constructWolfParams(QWidget*);
  QWidget* constructEvolutionParams(QWidget*);

  void fillGuiFromOptions();
  void fillOptionsFromGui();

  void startOptimization();

 private slots:
  void onIterationChanged(int iteration, double functional);

 private:
  optimization::GlobalOptions& options_;
  struct {
    std::string savePath;
    std::size_t iters;
  } copy_;

  std::future<Tensor<double, RepetitiveAllocator<double>>> optimResult_;
  Tensor<double, RepetitiveAllocator<double>> best_;
  std::vector<std::vector<double, RepetitiveAllocator<double>>> trajectory_;

  std::chrono::time_point<std::chrono::high_resolution_clock> tStart_;

  QLineEdit* tMax_;
  QLineEdit* dt_;
  QComboBox* method_;
  QLineEdit* seed_;
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
    QWidget* widget_;
  } wolf_;

  struct Evolution {
    QLineEdit* population_;
    QLineEdit* mutation_;
    QLineEdit* crossover_;
    QWidget* widget_;
  } evolution_;

  QPushButton* startOptimization_;
  QProgressBar* progress_;
  QLabel* iterations_;
  QLabel* iterTime_;
};
