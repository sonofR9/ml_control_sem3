#pragma once

#include "allocator.h"
#include "options.h"
#include "tensor.h"

#include <QMainWindow>

#include <chrono>
#include <future>
#include <memory>
#include <vector>

class QChart;
class QCheckBox;
class QComboBox;
class QCloseEvent;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QHBoxLayout;
class QVBoxLayout;
class QTableWidget;
class QWidget;

constexpr const char* kAppFolder{"ml_control_sem3"};
constexpr const char* kConfigPathFile{"/config_file_path.ini"};

class MainWindow : public QMainWindow {
  Q_OBJECT

  template <typename T, class Alloc = std::allocator<T>>
  using Tensor = optimization::Tensor<T, Alloc>;

  template <typename T>
  using Allocator = optimization::RepetitiveAllocator<T>;
  using DoubleAllocator = Allocator<double>;

 public:
  explicit MainWindow(optimization::GlobalOptions& options,
                      QWidget* parent = nullptr);

  ~MainWindow() override;

  void emitIterationChanged(std::size_t iteration, double functional);
  void emitBatchIterationChanged(std::size_t iteration, double functional);

 signals:
  void iterationChanged(int iteration, double functional);
  void batchIterationChanged(int iteration, double functional);
  void syncSharedWidgetsNow(int senderIndex);
  void closed();

 private:
  void constructView();

  QWidget* constructOptimizationTab(QWidget* tabWidget);
  QVBoxLayout* constructGlobalParams(QWidget*);
  QVBoxLayout* constructFunctionalParams(QWidget*);
  QVBoxLayout* constructControlParams(QWidget*);
  QWidget* constructWolfParams(QWidget*);
  QWidget* constructEvolutionParams(QWidget*);

  QWidget* constructShared(QWidget*);

  // info
  QWidget* constructInfoTab(QWidget* tabWidget);

  // for full screen chart
  QWidget* constructEmptyTab(QWidget* tabWidget);

  void closeEvent(QCloseEvent* event) override;

  void fillGuiFromOptions();
  void fillOptionsFromGui();

  void startOptimization();
  void startBatchOptimization();
  void startNextBatch();

  void gotResult();

  void enableCurrentOptimizationMethod();

 private slots:
  void onIterationChanged(int iteration, double functional);
  void onBatchIterationChanged(int iteration, double functional);

  void syncSharedWidgets(int senderIndex);

 private:
  optimization::GlobalOptions& options_;
  struct {
    std::string savePath;
    std::size_t iters;
    double tMax;
  } copy_;

  std::future<Tensor<double, DoubleAllocator>> optimResult_;
  Tensor<double, DoubleAllocator> best_;
  std::vector<std::vector<double, DoubleAllocator>> trajectory_;

  std::chrono::time_point<std::chrono::high_resolution_clock> tStart_;

  int batchCount_;
  int batchNumber_;

  // optimization tab
  QLineEdit* tMax_;
  QLineEdit* dt_;
  QComboBox* method_;
  QLineEdit* itersInput_;
  QLineEdit* seed_;
  QPushButton* saveFile_;
  QLineEdit* filePath_;
  QCheckBox* clear_;

  struct Functional {
    QLineEdit* coefTime_;
    QLineEdit* coefTerminal_;
    QLineEdit* coefObstacle_;
    QLineEdit* terminalTolerance_;
    // TODO(novak): circles
  } functional_;

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

  // info tab
  QTableWidget* bestDisplay_;

  // shared
  std::vector<QPushButton*> startOptimization_;
  std::vector<QPushButton*> startBatchOptimization_;
  std::vector<QLineEdit*> batchCountInput_;
  std::vector<QCheckBox*> updateOptionsDynamically_;

  std::vector<QProgressBar*> progress_;
  std::vector<QLabel*> iterations_;
  std::vector<QLabel*> iterTime_;

  std::vector<QChart*> charts_;
  std::vector<QLineEdit*> chartsDt_;
};
