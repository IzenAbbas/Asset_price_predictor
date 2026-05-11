import 'dart:convert';
import 'package:flutter/material.dart';
import '../models/prediction_model.dart';
import '../services/api_service.dart';

class EvaluationScreen extends StatefulWidget {
  const EvaluationScreen({super.key});

  @override
  State<EvaluationScreen> createState() => _EvaluationScreenState();
}

class _EvaluationScreenState extends State<EvaluationScreen>
    with SingleTickerProviderStateMixin {
  final _api = ApiService();
  late TabController _tabController;
  EvaluationOutput? _evaluation;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _loadEvaluation();
  }

  Future<void> _loadEvaluation() async {
    try {
      final eval = await _api.getEvaluationMetrics();
      setState(() {
        _evaluation = eval;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Widget _buildMetricsCard(EvaluationMetrics metrics, String title) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Divider(height: 32),
              _buildStatRow('Model', metrics.selectedModel),
              const SizedBox(height: 16),
              _buildStatRow('MAE', metrics.testMae.toStringAsFixed(2)),
              const SizedBox(height: 16),
              _buildStatRow('RMSE', metrics.testRmse.toStringAsFixed(2)),
              const SizedBox(height: 16),
              _buildStatRow('R²', metrics.testR2.toStringAsFixed(4)),
              const SizedBox(height: 24),
              const Text(
                'Graphs',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const Divider(),
              if (metrics.graphs.containsKey('predicted_vs_actual') &&
                  metrics.graphs['predicted_vs_actual']!.isNotEmpty) ...[
                const SizedBox(height: 16),
                const Text('Predicted vs. Actual', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Image.memory(base64Decode(metrics.graphs['predicted_vs_actual']!)),
              ],
              if (metrics.graphs.containsKey('residual_plot') &&
                  metrics.graphs['residual_plot']!.isNotEmpty) ...[
                const SizedBox(height: 16),
                const Text('Residual Plot', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Image.memory(base64Decode(metrics.graphs['residual_plot']!)),
              ],
              if (metrics.graphs.containsKey('learning_curve') &&
                  metrics.graphs['learning_curve']!.isNotEmpty) ...[
                const SizedBox(height: 16),
                const Text('Learning Curve', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Image.memory(base64Decode(metrics.graphs['learning_curve']!)),
              ],
              if (metrics.graphs.containsKey('feature_importance') &&
                  metrics.graphs['feature_importance']!.isNotEmpty) ...[
                const SizedBox(height: 16),
                const Text('Feature Importance', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Image.memory(base64Decode(metrics.graphs['feature_importance']!)),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
        ),
        Text(
          value,
          style: const TextStyle(fontSize: 16, color: Colors.blueGrey),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(child: Text(_error!));
    }
    if (_evaluation == null) {
      return const Center(child: Text('No evaluation data available.'));
    }

    return Column(
      children: [
        TabBar(
          controller: _tabController,
          labelColor: Theme.of(context).colorScheme.primary,
          unselectedLabelColor: Colors.grey,
          tabs: const [
            Tab(icon: Icon(Icons.directions_car), text: 'Car Predictor'),
            Tab(icon: Icon(Icons.home), text: 'House Predictor'),
          ],
        ),
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildMetricsCard(
                _evaluation!.carEvaluation,
                'Car Price Predictor Evaluation',
              ),
              _buildMetricsCard(
                _evaluation!.houseEvaluation,
                'House Price Predictor Evaluation',
              ),
            ],
          ),
        ),
      ],
    );
  }
}
