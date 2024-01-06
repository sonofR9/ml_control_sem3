#include "main-window.h"

#include "global.h"
#include "optimize-model.h"
#include "options.h"
#include "utils.h"

#include <QApplication>
#include <QChartView>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QLineSeries>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScatterSeries>
#include <QSettings>
#include <QStringList>
#include <QTabWidget>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QValidator>
#include <QWidget>

#include <array>
#include <fstream>
#include <random>

namespace {
using namespace optimization;
using Method = GlobalOptions::Method;
using EmitFunction = std::function<void(std::size_t, double)>;

constexpr int kSpacing{5};

void setTextIteration(std::vector<QLabel*> labels, int iter,
                      std::size_t maxIter, double functional) {
  for (auto label : labels) {
    label->setText(std::format("Iteration: {}/{} Functional: {:.5f}", iter,
                               maxIter, functional)
                       .c_str());
  }
}

void setTextTimePerIteration(std::vector<QLabel*> labels, double totalTime,
                             std::size_t maxIter) {
  if (maxIter != 0) {
    for (auto label : labels) {
      label->setText(std::format("Total time: {:6f}, Time per iteration: {:6f}",
                                 totalTime,
                                 totalTime / static_cast<double>(maxIter))
                         .c_str());
    }
  }
}

bool isFilePathValid(const std::string& path) {
  const std::ifstream test{path};
  return static_cast<bool>(test);
}

template <class Validator, class L = QVBoxLayout>
void addField(QWidget* parent, L* layout, const QString& lblName,
              QLineEdit*& result) {
  auto* hLayout{new QHBoxLayout{}};
  hLayout->setSpacing(kSpacing);
  layout->addItem(hLayout);

  auto* lbl{new QLabel{lblName, parent}};
  lbl->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  hLayout->addWidget(lbl);

  result = new QLineEdit{parent};
  result->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  result->setValidator(new Validator(result));
  hLayout->addWidget(result);
}

void setLineSeriesPen(QLineSeries* series, int width,
                      Qt::PenStyle style = Qt::SolidLine,
                      const QColor& color = Qt::blue) {
  QPen pen{series->pen()};
  pen.setStyle(style);
  pen.setColor(color);
  pen.setWidth(width);
  series->setPen(pen);
}

template <class Alloc>
void updateChart(QChart* chart, const std::vector<double, Alloc>& x,
                 const std::vector<double, Alloc>& y, double tolerance,
                 const std::vector<CircleData>& circles = {}) {
  QLineSeries* lastSeries{nullptr};
  if (!chart->series().empty()) {
    lastSeries = qobject_cast<QLineSeries*>(chart->series().last());
    lastSeries->setName("Previous");
    setLineSeriesPen(lastSeries, 2, Qt::DashLine, Qt::gray);
  }
  for (qsizetype i{chart->series().count() - 2}; i >= 0; --i) {
    QAbstractSeries* series = chart->series().at(i);
    chart->removeSeries(series);
  }

  for (const auto& circle : circles) {
    auto* circleSeries{new QLineSeries{}};
    for (auto i{0}; i <= 360; ++i) {
      double angle = M_PI * i / 180.0;
      double x = circle.x + circle.r * cos(angle);
      double y = circle.y + circle.r * sin(angle);
      circleSeries->append(x, y);
    }
    setLineSeriesPen(circleSeries, 2, Qt::SolidLine, Qt::red);
    chart->addSeries(circleSeries);
  }

  // auto* scatterSeries{new QScatterSeries{}};
  // scatterSeries->setName("Initial");
  // scatterSeries->append(10, 10);
  // scatterSeries->setMarkerSize(5);
  // scatterSeries->setMarkerShape(QScatterSeries::MarkerShapeTriangle);
  // scatterSeries->setColor(Qt::yellow);
  // chart->addSeries(scatterSeries);

  // add invisible point so that target will be always seen clearly
  auto* scatterSeries{new QScatterSeries{}};
  scatterSeries->append(-0.3, -0.3);
  scatterSeries->setMarkerSize(0.1);
  scatterSeries->setColor(Qt::white);
  scatterSeries->setBorderColor(Qt::white);
  chart->addSeries(scatterSeries);

  scatterSeries = new QScatterSeries{};
  scatterSeries->setName("Target");
  scatterSeries->append(0, 0);
  scatterSeries->setMarkerSize(20);
  scatterSeries->setMarkerShape(QScatterSeries::MarkerShapeStar);
  scatterSeries->setColor(Qt::green);
  scatterSeries->setBorderColor(Qt::green);
  chart->addSeries(scatterSeries);

  auto* newSeries{new QLineSeries()};
  newSeries->setName("Current");
  setLineSeriesPen(newSeries, 2, Qt::SolidLine, Qt::blue);
  for (size_t i = 0; i < x.size(); ++i) {
    newSeries->append(x[i], y[i]);
    if (std::sqrt(x[i] * x[i] + y[i] * y[i]) < tolerance) {
      break;
    }
  }

  chart->addSeries(newSeries);

  chart->createDefaultAxes();

  auto* chartView{qobject_cast<QChartView*>(chart->parentWidget())};
  if (chartView) {
    chartView->repaint();
  }
}

Tensor<Tensor<double, RepetitiveAllocator<double>>> reshapeTensor(
    const Tensor<double, RepetitiveAllocator<double>>& inputTensor, int step) {
  if (inputTensor.size() % step != 0) {
    throw std::invalid_argument("Input vector size must be divisible by step.");
  }
  int numRows{static_cast<int>(inputTensor.size()) / step};

  auto reshapedTensor = Tensor<Tensor<double, RepetitiveAllocator<double>>>(
      numRows, Tensor<double, RepetitiveAllocator<double>>(step));

  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < step; ++j) {
      reshapedTensor[i][j] = inputTensor[i * step + j];
    }
  }

  return reshapedTensor;
}

void refillTable(
    QTableWidget* table,
    const Tensor<Tensor<double, RepetitiveAllocator<double>>>& data,
    const QStringList& horizontalHeaders) {
  table->clear();
  table->setRowCount(static_cast<int>(data.size()));
  table->setColumnCount(static_cast<int>(data[0].size()));
  for (std::size_t row{0}; row < data.size(); ++row) {
    for (std::size_t column{0}; column < data[row].size(); ++column) {
      auto* item{new QTableWidgetItem(
          QString(std::format("{:.2f}", data[row][column]).c_str()),
          Qt::DisplayRole)};
      table->setItem(row, column, item);
    }
  }

  table->setHorizontalHeaderLabels(horizontalHeaders);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
}
}  // namespace

MainWindow::MainWindow(optimization::GlobalOptions& options, QWidget* parent)
    : QMainWindow(parent), options_{options} {
  restoreGeometry(
      QSettings(QDir::homePath() + "/" + kAppFolder + "/geometry.ini",
                QSettings::IniFormat)
          .value("main_window")
          .toByteArray());

  // batch count must be loaded before constructView
  batchCount_ = QSettings(QDir::homePath() + "/" + kAppFolder + "/misc.ini",
                          QSettings::IniFormat)
                    .value("batch_count")
                    .toInt();
  constructView();

  connect(this, &MainWindow::iterationChanged, this,
          &MainWindow::onIterationChanged);
  connect(this, &MainWindow::batchIterationChanged, this,
          &MainWindow::onBatchIterationChanged);

  if (options_.configFile.empty()) {
    QString path{};
    while (path.isEmpty()) {
      path = QFileDialog::getOpenFileName(
          this, "Open config", {},
          "Config files (*.ini);;Text files (*.txt);;All files (*.*)");
      if (path.isEmpty()) {
        QMessageBox::warning(this, "Error",
                             "No config file selected!\n"
                             "Please provide the config file.");
      }
    }
    options_.configFile = path.toStdString();
  }
  fillGuiFromOptions();
  centralWidget()->setEnabled(true);
  QSettings settings(QDir::homePath() + "/" + kAppFolder + kConfigPathFile,
                     QSettings::IniFormat);
  settings.setValue("config_file_path", QString(options_.configFile.c_str()));
}

MainWindow::~MainWindow() {
  QSettings(QDir::homePath() + "/" + kAppFolder + "/geometry.ini",
            QSettings::IniFormat)
      .setValue("main_window", saveGeometry());

  QSettings(QDir::homePath() + "/" + kAppFolder + "/misc.ini",
            QSettings::IniFormat)
      .setValue("batch_count", batchCountInput_[0]->text().toInt());
  QSettings(QDir::homePath() + "/" + kAppFolder + "/misc.ini",
            QSettings::IniFormat)
      .setValue("chart_dt", chartsDt_[0]->text().toDouble());

  fillOptionsFromGui();
  writeConfig(options_, options_.configFile);
  writeToSaveFile(options_.controlSaveFile, best_);
}

void MainWindow::startOptimization() {
  fillOptionsFromGui();
  copy_.iters = options_.iter;
  copy_.savePath = options_.controlSaveFile;
  copy_.tMax = options_.tMax;

  clear_->setCheckState(Qt::CheckState::Unchecked);

  for (auto* bar : progress_) {
    bar->setMaximum(static_cast<int>(copy_.iters));
    bar->setValue(0);
  }
  setTextIteration(iterations_, 0, copy_.iters, -1);

  tStart_ = std::chrono::high_resolution_clock::now();
  optimResult_ = std::async(
      std::launch::async, [this]() -> Tensor<double, DoubleAllocator> {
        auto printer = [this](std::size_t iteration, double functional) {
          emitIterationChanged(static_cast<int>(iteration), functional);
        };
        switch (options_.method) {
        case Method::kEvolution:
          return modelTestEvolution<Allocator, EmitFunction>(
              options_, EmitFunction(printer));
        case Method::kGrayWolf:
          return modelTestGray<Allocator, EmitFunction>(options_,
                                                        EmitFunction(printer));
        };
      });
  startOptimization_[0]->setEnabled(false);
  startBatchOptimization_[0]->setEnabled(false);
}

void MainWindow::onIterationChanged(int iteration, double functional) {
  for (auto* bar : progress_) {
    bar->setValue(iteration);
  }
  setTextIteration(iterations_, iteration, copy_.iters, functional);
  auto tNow = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(tNow - tStart_);
  setTextTimePerIteration(
      iterTime_, static_cast<double>(duration.count()) / 1000, iteration);

  if (iteration == static_cast<int>(copy_.iters)) {
    gotResult();
    startOptimization_[0]->setEnabled(true);
    startBatchOptimization_[0]->setEnabled(true);
  }
}

void MainWindow::startBatchOptimization() {
  fillOptionsFromGui();
  copy_.iters = options_.iter;
  copy_.savePath = options_.controlSaveFile;
  copy_.tMax = options_.tMax;

  batchNumber_ = 0;
  batchCount_ = batchCountInput_[0]->text().toInt();
  for (auto* bar : progress_) {
    bar->setMaximum(static_cast<int>(copy_.iters * batchCount_));
    bar->setValue(0);
  }
  setTextIteration(iterations_, 0, copy_.iters * batchCount_, -1);

  tStart_ = std::chrono::high_resolution_clock::now();
  startNextBatch();
  startOptimization_[0]->setEnabled(false);
  startBatchOptimization_[0]->setEnabled(false);
}

void MainWindow::startNextBatch() {
  ++batchNumber_;
  if (updateOptionsDynamically_[0]->isChecked()) {
    fillOptionsFromGui();
  } else {
    options_.clearSaveBeforeStart = clear_->isChecked();
  }

  optimResult_ = std::async(
      std::launch::async, [this]() -> Tensor<double, DoubleAllocator> {
        auto printer = [this](std::size_t iteration, double functional) {
          emitBatchIterationChanged(static_cast<int>(iteration), functional);
        };
        switch (options_.method) {
        case Method::kEvolution:
          return modelTestEvolution<Allocator, EmitFunction>(
              options_, EmitFunction(printer));
        case Method::kGrayWolf:
          return modelTestGray<Allocator, EmitFunction>(options_,
                                                        EmitFunction(printer));
        };
      });

  if (clear_->isChecked()) {
    clear_->setCheckState(Qt::CheckState::Unchecked);
  }
}

void MainWindow::onBatchIterationChanged(int iteration, double functional) {
  int adjustedIteration{iteration +
                        static_cast<int>(batchNumber_ * copy_.iters)};

  for (auto* bar : progress_) {
    bar->setValue(adjustedIteration);
  }
  setTextIteration(iterations_, adjustedIteration, copy_.iters * batchCount_,
                   functional);

  auto tNow = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(tNow - tStart_);
  setTextTimePerIteration(iterTime_,
                          static_cast<double>(duration.count()) / 1000,
                          adjustedIteration);

  if (iteration == static_cast<int>(copy_.iters)) {
    gotResult();

    if (adjustedIteration == static_cast<int>(copy_.iters * batchCount_)) {
      startOptimization_[0]->setEnabled(true);
      startBatchOptimization_[0]->setEnabled(true);
    } else {
      startNextBatch();
    }
  }
}

void MainWindow::gotResult() {
  best_ = optimResult_.get();
  writeToSaveFile(options_.controlSaveFile, best_);
  refillTable(bestDisplay_, reshapeTensor(best_, 2), {"u1", "u2"});

  trajectory_ =
      two_wheeled_robot::getTrajectoryFromControl<double, DoubleAllocator>(
          best_, options_.tMax, chartsDt_[0]->text().toDouble());
  for (const auto& chart : charts_) {
    updateChart(chart, trajectory_[0], trajectory_[1],
                options_.functionalOptions.terminalTolerance,
                {{.x = 2.5, .y = 2.5, .r = std::sqrt(2.5)},
                 {.x = 7.5, .y = 7.5, .r = std::sqrt(2.5)}});
  }
}

void MainWindow::emitIterationChanged(std::size_t iteration,
                                      double functional) {
  emit iterationChanged(static_cast<int>(iteration), functional);
}

void MainWindow::emitBatchIterationChanged(std::size_t iteration,
                                           double functional) {
  emit batchIterationChanged(static_cast<int>(iteration), functional);
}

void MainWindow::constructView() {
  setWindowFlags(windowFlags());
  setWindowTitle("ml control sem3");
  auto* central{new QWidget{this}};
  setCentralWidget(central);
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setSpacing(kSpacing);
  centralWidget()->setLayout(vLayout);

  auto* tabWidget{new QTabWidget{this}};
  tabWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  vLayout->addWidget(tabWidget);

  auto* optimizationTab{constructOptimizationTab(tabWidget)};
  tabWidget->addTab(optimizationTab, "Optimization");
  auto* infoTab{constructInfoTab(tabWidget)};
  tabWidget->addTab(infoTab, "Info");
  auto* chartTab{constructEmptyTab(tabWidget)};
  tabWidget->addTab(chartTab, "Full screen chart");

  for (int i{0}; i < tabWidget->count(); ++i) {
    auto* tab{tabWidget->widget(i)};
    tab->layout()->addWidget(constructShared(tab));
  }

  auto* menu{menuBar()};
  auto* config{menu->addMenu("&Config")};
  auto* openConfig{config->addAction("Open config...")};
  connect(openConfig, &QAction::triggered, [this]() -> void {
    const auto path{QFileDialog::getOpenFileName(
        this, "Open config", {},
        "Config files (*.ini);;Text files (*.txt);;All files (*.*)")};
    if (path.isEmpty()) {
      return;
    }
    options_.configFile = path.toStdString();
    fillGuiFromOptions();
    centralWidget()->setEnabled(true);
    QSettings settings(QDir::homePath() + "/" + kAppFolder + kConfigPathFile,
                       QSettings::IniFormat);
    settings.setValue("config_file_path", QString(path));
  });

  auto* chartMenu{menu->addMenu("&Chart")};
  auto* chartInNewWindow{chartMenu->addAction("Open in new window")};
  connect(chartInNewWindow, &QAction::triggered, [this]() -> void {
    auto* dialog{new QDialog()};
    dialog->setWindowFlags(Qt::Window);  // Qt::WindowMinMaxButtonsHint |
                                         // Qt::WindowCloseButtonHint
    auto* vLayout{new QVBoxLayout{dialog}};
    auto* chartView{new QChartView{dialog}};
    vLayout->addWidget(chartView);

    auto* chart{chartView->chart()};
    charts_.push_back(chart);

    connect(dialog, &QDialog::finished, [this, chart]() {
      std::erase_if(charts_, [chart](QChart* c) -> bool { return c == chart; });
    });

    connect(this, &MainWindow::closed, [dialog]() {
      dialog->disconnect();
      dialog->close();
    });

    dialog->show();
  });
}

QWidget* MainWindow::constructOptimizationTab(QWidget* tabWidget) {
  auto* tab{new QWidget{tabWidget}};
  auto* vLayout{new QVBoxLayout{}};
  tab->setLayout(vLayout);

  auto* hLayout{new QHBoxLayout{}};
  vLayout->addItem(hLayout);
  auto* globalParams{constructGlobalParams(tab)};
  hLayout->addItem(globalParams);
  auto* functionalParams{constructFunctionalParams(tab)};
  hLayout->addItem(functionalParams);
  auto* controlParams{constructControlParams(tab)};
  hLayout->addItem(controlParams);
  auto* wolfParams{constructWolfParams(tab)};
  hLayout->addWidget(wolfParams);
  auto* evolutionParams{constructEvolutionParams(tab)};
  hLayout->addWidget(evolutionParams);
  enableCurrentOptimizationMethod();

  hLayout->addStretch(1);
  hLayout->setSpacing(kSpacing);

  vLayout->setSpacing(kSpacing);

  return tab;
}

QVBoxLayout* MainWindow::constructGlobalParams(QWidget* tab) {
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setAlignment(Qt::AlignTop);
  vLayout->setSpacing(kSpacing);

  auto* title{new QLabel{"Global parameters", tab}};
  title->setAlignment(Qt::AlignmentFlag::AlignCenter);
  vLayout->addWidget(title);

  addField<QDoubleValidator>(tab, vLayout, "tmax", tMax_);
  addField<QDoubleValidator>(tab, vLayout, "dt", dt_);
  addField<QIntValidator>(tab, vLayout, "iterations", itersInput_);

  auto* hLayout{new QHBoxLayout{}};
  hLayout->setSpacing(kSpacing);
  vLayout->addItem(hLayout);
  auto* lbl{new QLabel{"method", tab}};
  hLayout->addWidget(lbl);
  method_ = new QComboBox{tab};
  method_->insertItem(static_cast<int>(Method::kGrayWolf),
                      methodToName(Method::kGrayWolf).c_str());
  method_->insertItem(static_cast<int>(Method::kEvolution),
                      methodToName(Method::kEvolution).c_str());
  connect(method_, qOverload<int>(&QComboBox::currentIndexChanged),
          [this](int) { enableCurrentOptimizationMethod(); });
  hLayout->addWidget(method_);

  addField<QIntValidator>(tab, vLayout, "seed (-1 random)", seed_);
  auto* resetSeed{new QPushButton{"Set seed", tab}};
  connect(resetSeed, &QPushButton::clicked, [this]() {
    const auto seed{seed_->text().toInt()};
    if (seed < 0) {
      SharedGenerator::gen.seed(std::random_device{}());
    } else {
      SharedGenerator::gen.seed(seed_->text().toUInt());
    }
  });
  vLayout->addWidget(resetSeed);

  hLayout = new QHBoxLayout{};
  hLayout->setSpacing(kSpacing);
  vLayout->addItem(hLayout);
  saveFile_ = new QPushButton{"Save-file", tab};
  connect(saveFile_, &QPushButton::clicked, [this]() -> void {
    // open file dialog
    const auto fileName = QFileDialog::getOpenFileName(this, "Choose save file",
                                                       QDir::homePath());
    if (fileName.isEmpty()) {
      return;
    }
    filePath_->setText(fileName);
    options_.controlSaveFile = fileName.toStdString();
  });
  hLayout->addWidget(saveFile_);
  filePath_ = new QLineEdit{tab};
  connect(filePath_, &QLineEdit::returnPressed, [this]() {
    if (isFilePathValid(filePath_->text().toStdString())) {
      options_.controlSaveFile = filePath_->text().toStdString();
    } else {
      // show error message
      const auto revert = QMessageBox::warning(
          this, "Error",
          QString("Invalid file path: %1. \nRevert to "
                  "last saved (if not you will need to correct path)?")
              .arg(filePath_->text()),
          QMessageBox::Yes | QMessageBox::No);
      if (revert == QMessageBox::Yes) {
        filePath_->setText(options_.controlSaveFile.c_str());
      } else {
        filePath_->setFocus();
      }
    }
  });
  hLayout->addWidget(filePath_);

  clear_ = new QCheckBox{"Start from scratch (clear save)", tab};
  vLayout->addWidget(clear_);

  return vLayout;
}

QVBoxLayout* MainWindow::constructFunctionalParams(QWidget* tab) {
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setAlignment(Qt::AlignTop);
  vLayout->setSpacing(kSpacing);

  auto* title{new QLabel{"Functional parameters", tab}};
  title->setAlignment(Qt::AlignmentFlag::AlignCenter);
  title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  vLayout->addWidget(title);

  auto* subTitleCoefficients{new QLabel{"Coefficients", tab}};
  subTitleCoefficients->setAlignment(Qt::AlignmentFlag::AlignCenter);
  subTitleCoefficients->setSizePolicy(QSizePolicy::Preferred,
                                      QSizePolicy::Maximum);
  auto font{subTitleCoefficients->font()};
  font.setItalic(true);
  subTitleCoefficients->setFont(font);
  vLayout->addWidget(subTitleCoefficients);

  addField<QDoubleValidator>(tab, vLayout, "time", functional_.coefTime_);
  addField<QDoubleValidator>(tab, vLayout, "terminal position",
                             functional_.coefTerminal_);
  addField<QDoubleValidator>(tab, vLayout, "obstacle",
                             functional_.coefObstacle_);

  auto* subTitleOther{new QLabel{"Other", tab}};
  subTitleOther->setAlignment(Qt::AlignmentFlag::AlignCenter);
  subTitleOther->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  font = subTitleOther->font();
  font.setItalic(true);
  subTitleOther->setFont(font);
  vLayout->addWidget(subTitleOther);

  addField<QDoubleValidator>(tab, vLayout, "terminal tolerance",
                             functional_.terminalTolerance_);

  return vLayout;
}

QVBoxLayout* MainWindow::constructControlParams(QWidget* tab) {
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setAlignment(Qt::AlignTop);
  vLayout->setSpacing(kSpacing);

  auto* title{new QLabel{"Control parameters", tab}};
  title->setAlignment(Qt::AlignmentFlag::AlignCenter);
  title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  vLayout->addWidget(title);

  addField<QIntValidator>(tab, vLayout, "number", control_.number_);
  addField<QDoubleValidator>(tab, vLayout, "min", control_.min_);
  addField<QDoubleValidator>(tab, vLayout, "max", control_.max_);

  return vLayout;
}

QWidget* MainWindow::constructWolfParams(QWidget* tab) {
  wolf_.widget_ = new QWidget{tab};
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setAlignment(Qt::AlignTop);
  vLayout->setSpacing(kSpacing);
  wolf_.widget_->setLayout(vLayout);

  auto* title{new QLabel{"Wolf parameters", wolf_.widget_}};
  title->setAlignment(Qt::AlignmentFlag::AlignCenter);
  title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  vLayout->addWidget(title);

  addField<QIntValidator>(wolf_.widget_, vLayout, "number", wolf_.num_);
  addField<QIntValidator>(wolf_.widget_, vLayout, "Best number", wolf_.best_);

  return wolf_.widget_;
}

QWidget* MainWindow::constructEvolutionParams(QWidget* tab) {
  evolution_.widget_ = new QWidget{tab};
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setAlignment(Qt::AlignTop);
  vLayout->setSpacing(kSpacing);
  evolution_.widget_->setLayout(vLayout);

  auto* title{new QLabel{"Evolution parameters", evolution_.widget_}};
  title->setAlignment(Qt::AlignmentFlag::AlignCenter);
  title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  vLayout->addWidget(title);

  addField<QIntValidator>(evolution_.widget_, vLayout, "population",
                          evolution_.population_);
  addField<QDoubleValidator>(evolution_.widget_, vLayout, "mutation rate",
                             evolution_.mutation_);
  addField<QDoubleValidator>(evolution_.widget_, vLayout, "crossover rate",
                             evolution_.crossover_);

  return evolution_.widget_;
}

QWidget* MainWindow::constructShared(QWidget* tab) {
  int index{static_cast<int>(startOptimization_.size())};

  auto* shared{new QWidget{tab}};
  auto* vLayout{new QVBoxLayout{}};
  shared->setLayout(vLayout);
  vLayout->setSpacing(kSpacing);

  auto* hLayout{new QHBoxLayout{}};
  hLayout->setSpacing(kSpacing);

  startOptimization_.emplace_back(
      new QPushButton{"Start optimization", shared});
  connect(startOptimization_.back(), &QPushButton::clicked, [this, index]() {
    startOptimization();
    syncSharedInputs(index);
  });
  hLayout->addWidget(startOptimization_.back());

  hLayout->addStretch(1);

  startBatchOptimization_.emplace_back(
      new QPushButton{"Start batch optimization", shared});
  hLayout->addWidget(startBatchOptimization_.back());
  connect(startBatchOptimization_.back(), &QPushButton::clicked,
          [this, index]() {
            startBatchOptimization();
            syncSharedInputs(index);
          });

  batchCountInput_.emplace_back(nullptr);
  addField<QIntValidator, QHBoxLayout>(shared, hLayout, "batch count",
                                       batchCountInput_.back());
  batchCountInput_.back()->setText(QString::number(batchCount_));
  connect(batchCountInput_.back(), &QLineEdit::editingFinished,
          [this, index]() { syncSharedInputs(index); });

  updateOptionsDynamically_.emplace_back(
      new QCheckBox{"Update options dynamically", shared});
  connect(updateOptionsDynamically_.back(), &QCheckBox::stateChanged,
          [this, index]() { syncSharedInputs(index); });
  hLayout->addWidget(updateOptionsDynamically_.back());

  vLayout->addItem(hLayout);

  progress_.emplace_back(new QProgressBar{shared});
  vLayout->addWidget(progress_.back());

  hLayout = new QHBoxLayout{};
  vLayout->addItem(hLayout);
  iterations_.emplace_back(new QLabel{shared});
  setTextIteration(iterations_, 0, 0, 0);
  hLayout->addWidget(iterations_.back());
  iterTime_.emplace_back(new QLabel{shared});
  iterTime_.back()->setText("Run to display time statistics");
  iterTime_.back()->setAlignment(Qt::AlignmentFlag::AlignRight);
  hLayout->addWidget(iterTime_.back());

  hLayout = new QHBoxLayout();
  auto* chartView{new QChartView{shared}};
  charts_.push_back(chartView->chart());
  chartView->setRenderHint(QPainter::Antialiasing);
  chartView->setSizePolicy(QSizePolicy::Preferred,
                           QSizePolicy::MinimumExpanding);
  hLayout->addWidget(chartView);
  hLayout->addItem(
      new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding));
  vLayout->addItem(hLayout);

  chartsDt_.emplace_back(nullptr);
  addField<QDoubleValidator>(
      shared, vLayout, "Display trajectory integration dt", chartsDt_.back());
  chartsDt_.back()->setText(QString::number(
      QSettings(QDir::homePath() + "/" + kAppFolder + "/misc.ini",
                QSettings::IniFormat)
          .value("chart_dt")
          .toDouble()));
  connect(chartsDt_.back(), &QLineEdit::editingFinished,
          [this, index]() { syncSharedInputs(index); });

  return shared;
}

QWidget* MainWindow::constructInfoTab(QWidget* tabWidget) {
  auto* tab{new QWidget{tabWidget}};
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setSpacing(kSpacing);
  tab->setLayout(vLayout);

  auto* hLayout{new QHBoxLayout{}};
  hLayout->setSpacing(kSpacing);
  vLayout->addItem(hLayout);

  bestDisplay_ = new QTableWidget{tab};
  bestDisplay_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  hLayout->addWidget(bestDisplay_);

  hLayout->addStretch(1);

  return tab;
}

QWidget* MainWindow::constructEmptyTab(QWidget* tabWidget) {
  auto* tab{new QWidget{tabWidget}};
  auto* vLayout{new QVBoxLayout{}};
  vLayout->setSpacing(kSpacing);
  tab->setLayout(vLayout);

  return tab;
}

void MainWindow::closeEvent(QCloseEvent* event) {
  emit closed();
  QMainWindow::closeEvent(event);
}

void MainWindow::fillGuiFromOptions() {
  // Global options
  tMax_->setText(QString::number(options_.tMax));
  dt_->setText(QString::number(options_.integrationDt));
  itersInput_->setText(QString::number(options_.iter));

  // functional
  functional_.coefTime_->setText(
      QString::number(options_.functionalOptions.coefTime));
  functional_.coefTerminal_->setText(
      QString::number(options_.functionalOptions.coefTerminal));
  functional_.coefObstacle_->setText(
      QString::number(options_.functionalOptions.coefObstacle));
  functional_.terminalTolerance_->setText(
      QString::number(options_.functionalOptions.terminalTolerance));
  // TODO(novak) add circles

  // Set optimization method
  method_->setCurrentIndex(static_cast<int>(options_.method));

  seed_->setText(QString::number(options_.seed));
  filePath_->setText(QString::fromStdString(options_.controlSaveFile));
  clear_->setChecked(options_.clearSaveBeforeStart);

  // Control options
  control_.number_->setText(
      QString::number(options_.controlOptions.numOfParams));
  control_.min_->setText(QString::number(options_.controlOptions.uMin));
  control_.max_->setText(QString::number(options_.controlOptions.uMax));

  // Gray Wolf options
  wolf_.num_->setText(QString::number(options_.wolfOpt.wolfNum));
  wolf_.best_->setText(QString::number(options_.wolfOpt.numBest));

  // Evolution options
  evolution_.population_->setText(
      QString::number(options_.evolutionOpt.populationSize));
  evolution_.mutation_->setText(
      QString::number(options_.evolutionOpt.mutationRate));
  evolution_.crossover_->setText(
      QString::number(options_.evolutionOpt.crossoverRate));
}

void MainWindow::fillOptionsFromGui() {
  // Global options
  options_.tMax = tMax_->text().toDouble();
  options_.integrationDt = dt_->text().toDouble();
  options_.iter = std::max(itersInput_->text().toUInt(), 1U);

  // functional options
  options_.functionalOptions.coefTime =
      functional_.coefTime_->text().toDouble();
  options_.functionalOptions.coefTerminal =
      functional_.coefTerminal_->text().toDouble();
  options_.functionalOptions.coefObstacle =
      functional_.coefObstacle_->text().toDouble();
  options_.functionalOptions.terminalTolerance =
      functional_.terminalTolerance_->text().toDouble();
  // TODO(novak) circles

  // Read optimization method
  options_.method = static_cast<Method>(method_->currentIndex());

  const auto seed{seed_->text().toInt()};
  if (seed < 0) {
    options_.seed = std::random_device{}();
  } else {
    options_.seed = seed_->text().toUInt();
  }
  options_.controlSaveFile = filePath_->text().toStdString();
  options_.clearSaveBeforeStart = clear_->isChecked();

  // Control options
  options_.controlOptions.numOfParams = control_.number_->text().toULongLong();
  options_.controlOptions.uMin = control_.min_->text().toDouble();
  options_.controlOptions.uMax = control_.max_->text().toDouble();

  // Gray Wolf options
  options_.wolfOpt.wolfNum = wolf_.num_->text().toInt();
  options_.wolfOpt.numBest = wolf_.best_->text().toInt();

  // Evolution options
  options_.evolutionOpt.populationSize = evolution_.population_->text().toInt();
  options_.evolutionOpt.mutationRate = evolution_.mutation_->text().toDouble();
  options_.evolutionOpt.crossoverRate =
      evolution_.crossover_->text().toDouble();
}

void MainWindow::enableCurrentOptimizationMethod() {
  evolution_.widget_->hide();
  wolf_.widget_->hide();
  switch (static_cast<Method>(method_->currentIndex())) {
  case Method::kEvolution:
    evolution_.widget_->show();
    break;
  case Method::kGrayWolf:
    wolf_.widget_->show();
    break;
  }
}

void MainWindow::syncSharedInputs(int senderIndex) {
  const bool startEnabled{startOptimization_[senderIndex]->isEnabled()};
  const auto batchCount{batchCountInput_[senderIndex]->text()};
  const bool updateDynamically{
      updateOptionsDynamically_[senderIndex]->isChecked()};
  const auto chartDt{chartsDt_[senderIndex]->text()};

  for (std::size_t i{0}; i < startOptimization_.size(); ++i) {
    startOptimization_[i]->blockSignals(true);
    startBatchOptimization_[i]->blockSignals(true);
    batchCountInput_[i]->blockSignals(true);
    updateOptionsDynamically_[i]->blockSignals(true);
    chartsDt_[i]->blockSignals(true);

    startOptimization_[i]->setEnabled(startEnabled);
    startBatchOptimization_[i]->setEnabled(startEnabled);
    batchCountInput_[i]->setText(batchCount);
    updateOptionsDynamically_[i]->setChecked(updateDynamically);
    chartsDt_[i]->setText(chartDt);

    startOptimization_[i]->blockSignals(false);
    startBatchOptimization_[i]->blockSignals(false);
    batchCountInput_[i]->blockSignals(false);
    updateOptionsDynamically_[i]->blockSignals(false);
    chartsDt_[i]->blockSignals(false);
  }
}
