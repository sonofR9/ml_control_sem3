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
#include <QLabel>
#include <QLineEdit>
#include <QLineSeries>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QValidator>
#include <QWidget>

#include <array>
#include <fstream>
#include <random>

namespace {
using namespace optimization;
using Method = GlobalOptions::Method;

constexpr int kSpacing{5};

void setTextIteration(QLabel* label, int iter, std::size_t maxIter,
                      double functional) {
  label->setText(std::format("Iteration: {}/{} Functional: {:.5f}", iter,
                             maxIter, functional)
                     .c_str());
}

void setTextTimePerIteration(QLabel* label, double totalTime,
                             std::size_t maxIter) {
  if (maxIter != 0) {
    label->setText(std::format("Total time: {:6f}, Time per iteration: {:6f}",
                               totalTime,
                               totalTime / static_cast<double>(maxIter))
                       .c_str());
  }
}

bool isFilePathValid(const std::string& path) {
  const std::ifstream test{path};
  return static_cast<bool>(test);
}

template <class Validator>
void addField(QWidget* parent, QVBoxLayout* vLayout, const QString& lblName,
              QLineEdit*& result) {
  auto* hLayout{new QHBoxLayout{}};
  hLayout->setSpacing(kSpacing);
  vLayout->addItem(hLayout);

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
                 const std::vector<double, Alloc>& y,
                 const std::vector<CircleData> circles = {}) {
  QLineSeries* lastSeries{nullptr};
  if (!chart->series().empty()) {
    lastSeries = qobject_cast<QLineSeries*>(chart->series().last());
    lastSeries->setName("Previous");
    setLineSeriesPen(lastSeries, 3, Qt::DashLine, Qt::gray);
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
    setLineSeriesPen(circleSeries, 3, Qt::SolidLine, Qt::red);
    chart->addSeries(circleSeries);
  }

  auto* newSeries{new QLineSeries()};
  newSeries->setName("Current");
  setLineSeriesPen(newSeries, 3, Qt::SolidLine, Qt::blue);
  for (size_t i = 0; i < x.size(); ++i) {
    newSeries->append(x[i], y[i]);
  }

  chart->addSeries(newSeries);

  chart->createDefaultAxes();

  auto* chartView{qobject_cast<QChartView*>(chart->parentWidget())};
  if (chartView) {
    chartView->repaint();
  }
}
}  // namespace

MainWindow::MainWindow(optimization::GlobalOptions& options, QWidget* parent)
    : QMainWindow(parent), options_{options} {
  restoreGeometry(
      QSettings(QDir::homePath() + "/" + kAppFolder + "/geometry.ini",
                QSettings::IniFormat)
          .value("main_window")
          .toByteArray());
  constructView();
  connect(this, &MainWindow::iterationChanged, this,
          &MainWindow::onIterationChanged);

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
  fillOptionsFromGui();
  writeConfig(options_, options_.configFile);
  writeToSaveFile(options_.controlSaveFile, best_);
}

void MainWindow::startOptimization() {
  fillOptionsFromGui();
  copy_.iters = options_.iter;
  copy_.savePath = options_.controlSaveFile;
  copy_.tMax = options_.tMax;

  progress_->setMaximum(static_cast<int>(copy_.iters));
  progress_->setValue(0);
  setTextIteration(iterations_, 0, copy_.iters, -1);

  tStart_ = std::chrono::high_resolution_clock::now();
  optimResult_ = std::async(
      std::launch::async, [this]() -> Tensor<double, DoubleAllocator> {
        auto printer = [this](int iteration, double functional) {
          emitIterationChanged(iteration, functional);
        };
        switch (options_.method) {
        case Method::kEvolution:
          return modelTestEvolution<Allocator, decltype(printer)>(options_,
                                                                  printer);
        case Method::kGrayWolf:
          return modelTestGray<Allocator, decltype(printer)>(options_, printer);
        };
      });
  startOptimization_->setEnabled(false);
  startOptimization_->show();
}

void MainWindow::onIterationChanged(int iteration, double functional) {
  progress_->setValue(iteration);
  setTextIteration(iterations_, iteration, copy_.iters, functional);
  auto tNow = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(tNow - tStart_);
  setTextTimePerIteration(
      iterTime_, static_cast<double>(duration.count()) / 1000, iteration);

  if (iteration == static_cast<int>(copy_.iters)) {
    best_ = optimResult_.get();
    writeToSaveFile(options_.controlSaveFile, best_);
    startOptimization_->setEnabled(true);
    startOptimization_->show();
    trajectory_ =
        two_wheeled_robot::getTrajectoryFromControl<double, DoubleAllocator>(
            best_, copy_.tMax);
    for (const auto& chart : charts_) {
      updateChart(chart, trajectory_[0], trajectory_[1],
                  {{.x = 2.5, .y = 2.5, .r = std::sqrt(2.5)},
                   {.x = 7.5, .y = 7.5, .r = std::sqrt(2.5)}});
    }
  }
}

void MainWindow::emitIterationChanged(std::size_t iteration,
                                      double functional) {
  emit iterationChanged(static_cast<int>(iteration), functional);
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

  auto* chartView = new QChartView{centralWidget()};
  charts_.push_back(chartView->chart());
  chartView->setRenderHint(QPainter::Antialiasing);
  chartView->setSizePolicy(QSizePolicy::Preferred,
                           QSizePolicy::MinimumExpanding);
  vLayout->addWidget(chartView);

  // does not update when tab 1 is selected
  // auto* chartTab{new QChartView{tabWidget}};
  // charts_.push_back(chartTab->chart());
  // chartTab->setRenderHint(QPainter::Antialiasing);
  // chartTab->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  // tabWidget->addTab(chartTab, "Full screen chart");
  // connect(tabWidget, &QTabWidget::currentChanged,
  //         [this, tabWidget, chartView](int index) -> void {
  //           if (index == 0) {
  //             chartView->show();
  //             chartView->setSizePolicy(QSizePolicy::Preferred,
  //                                      QSizePolicy::MinimumExpanding);
  //             chartView->updateGeometry();
  //             tabWidget->updateGeometry();
  //           } else {
  //             // chartTab->setChart(charts_);
  //             chartView->hide();
  //           }
  //           QApplication::processEvents();
  //         });

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
  auto* controlParams{constructControlParams(tab)};
  hLayout->addItem(controlParams);
  auto* wolfParams{constructWolfParams(tab)};
  hLayout->addWidget(wolfParams);
  auto* evolutionParams{constructEvolutionParams(tab)};
  hLayout->addWidget(evolutionParams);
  enableCurrentOptimizationMethod();

  hLayout->addStretch(1);
  hLayout->setSpacing(kSpacing);

  startOptimization_ = new QPushButton{"Start optimization", tab};
  vLayout->addWidget(startOptimization_);
  connect(startOptimization_, &QPushButton::clicked, [this]() {
    startOptimization_->hide();
    progress_->show();
    iterations_->show();
    iterTime_->show();
    startOptimization();
  });

  progress_ = new QProgressBar{tab};
  vLayout->addWidget(progress_);

  hLayout = new QHBoxLayout{};
  vLayout->addItem(hLayout);
  iterations_ = new QLabel{tab};
  setTextIteration(iterations_, 0, 0, 0);
  hLayout->addWidget(iterations_);
  iterTime_ = new QLabel{tab};
  iterTime_->setText("Run to display time statistics");
  iterTime_->setAlignment(Qt::AlignmentFlag::AlignRight);
  hLayout->addWidget(iterTime_);

  vLayout->setSpacing(kSpacing);

  return tab;
}

QVBoxLayout* MainWindow::constructGlobalParams(QWidget* tab) {
  auto* vLayout{new QVBoxLayout{}};
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

  vLayout->addStretch(1);
  vLayout->setSpacing(kSpacing);

  return vLayout;
}

QVBoxLayout* MainWindow::constructControlParams(QWidget* tab) {
  auto* vLayout{new QVBoxLayout{}};
  auto* title{new QLabel{"Control parameters", tab}};
  title->setAlignment(Qt::AlignmentFlag::AlignCenter);
  title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  vLayout->addWidget(title);

  addField<QIntValidator>(tab, vLayout, "number", control_.number_);
  addField<QDoubleValidator>(tab, vLayout, "min", control_.min_);
  addField<QDoubleValidator>(tab, vLayout, "max", control_.max_);

  vLayout->addStretch(1);
  vLayout->setSpacing(kSpacing);

  return vLayout;
}

QWidget* MainWindow::constructWolfParams(QWidget* tab) {
  wolf_.widget_ = new QWidget{tab};
  auto* vLayout{new QVBoxLayout{}};
  wolf_.widget_->setLayout(vLayout);
  auto* title{new QLabel{"Wolf parameters", wolf_.widget_}};
  title->setAlignment(Qt::AlignmentFlag::AlignCenter);
  title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  vLayout->addWidget(title);

  addField<QIntValidator>(wolf_.widget_, vLayout, "number", wolf_.num_);
  addField<QIntValidator>(wolf_.widget_, vLayout, "Best number", wolf_.best_);

  vLayout->addStretch(1);
  vLayout->setSpacing(kSpacing);

  return wolf_.widget_;
}

QWidget* MainWindow::constructEvolutionParams(QWidget* tab) {
  evolution_.widget_ = new QWidget{tab};
  auto* vLayout{new QVBoxLayout{}};
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

  vLayout->addStretch(1);
  vLayout->setSpacing(kSpacing);

  return evolution_.widget_;
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
