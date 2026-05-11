import 'dart:convert';

import 'package:flutter/foundation.dart';
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

  Widget _buildMetricsCard(
    EvaluationMetrics metrics,
    String title, {
    required bool isHouse,
  }) {
    final metricInsights = isHouse ? _houseMetricInsights : _carMetricInsights;
    final graphInsights = isHouse ? _houseGraphInsights : _carGraphInsights;

    final screenWidth = MediaQuery.of(context).size.width;
    final isWideWeb = kIsWeb && screenWidth >= 900;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Card(
            elevation: 8,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const Divider(height: 32),
                  _buildStatRow('Model', metrics.selectedModel),
                  const SizedBox(height: 12),
                  _buildExpandableStatRow(
                    label: 'MAE',
                    value: metrics.testMae.toStringAsFixed(2),
                    insight: metricInsights['mae'] ?? '',
                  ),
                  const SizedBox(height: 12),
                  _buildExpandableStatRow(
                    label: 'RMSE',
                    value: metrics.testRmse.toStringAsFixed(2),
                    insight: metricInsights['rmse'] ?? '',
                  ),
                  const SizedBox(height: 12),
                  _buildExpandableStatRow(
                    label: 'R²',
                    value: metrics.testR2.toStringAsFixed(4),
                    insight: metricInsights['r2'] ?? '',
                  ),
                ],
              ),
            ),
          ),
          if (metrics.graphs.isNotEmpty) ...[
            const SizedBox(height: 24),
            Text(
              'Graphs',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            ...metrics.graphs.entries.map(
              (entry) => Padding(
                padding: const EdgeInsets.only(bottom: 20),
                child: _buildGraphCard(
                  title: _formatGraphTitle(entry.key),
                  base64Image: entry.value,
                  insight:
                      graphInsights[entry.key] ??
                      'No insight is available for this graph yet.',
                  isWideWeb: isWideWeb,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildExpandableStatRow({
    required String label,
    required String value,
    required String insight,
  }) {
    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        tilePadding: EdgeInsets.zero,
        childrenPadding: const EdgeInsets.only(top: 8),
        shape: const RoundedRectangleBorder(side: BorderSide.none),
        collapsedShape: const RoundedRectangleBorder(side: BorderSide.none),
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
            Text(
              value,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
        children: [
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              insight,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurface,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGraphCard({
    required String title,
    required String base64Image,
    required String insight,
    required bool isWideWeb,
  }) {
    final card = Card(
      elevation: 6,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          shape: const RoundedRectangleBorder(side: BorderSide.none),
          collapsedShape: const RoundedRectangleBorder(side: BorderSide.none),
          title: Text(
            title,
            style: Theme.of(
              context,
            ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
          ),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 12),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.memory(
                base64Decode(base64Image),
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) {
                  return Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    color: Theme.of(context).colorScheme.surfaceVariant,
                    child: Text(
                      'Could not render $title graph.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                insight,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurface,
                ),
              ),
            ),
          ],
        ),
      ),
    );

    if (!isWideWeb) {
      return card;
    }

    return Align(
      alignment: Alignment.topCenter,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 760),
        child: card,
      ),
    );
  }

  String _formatGraphTitle(String key) {
    return key
        .replaceAll('_', ' ')
        .replaceFirstMapped(
          RegExp(r'^[a-z]'),
          (match) => match.group(0)!.toUpperCase(),
        );
  }

  Widget _buildStatRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: Theme.of(
            context,
          ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
        ),
        Text(
          value,
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }

  static const Map<String, String> _houseMetricInsights = {
    'mae':
        'On average, the model\'s guess for a house price is off by about 4.6 million.',
    'rmse':
        'A few extreme mistakes in guessing push the overall error penalty up to roughly 15.3 million.',
    'r2':
        'The model successfully understands about 81% of the reasons why house prices change.',
  };

  static const Map<String, String> _houseGraphInsights = {
    'predicted_vs_actual':
        'The model is fairly accurate for average-priced houses but struggles to guess the right price for very expensive ones.',
    'residual_plot':
        'The more expensive the house actually is, the less accurate the model\'s guess tends to be.',
    'feature_importance':
        'The total size of the house and its specific location are the biggest clues the model uses to guess the price.',
    'learning_curve':
        'Providing the model with more examples is steadily helping it learn and make fewer mistakes on new data.',
  };

  static const Map<String, String> _carMetricInsights = {
    'mae':
        'On average, the model\'s guess for a car price is off by about 441,000.',
    'rmse':
        'A handful of bad guesses raises the overall error penalty to around 1.46 million.',
    'r2':
        'The model is highly reliable, understanding almost 94% of what determines a car\'s value.',
  };

  static const Map<String, String> _carGraphInsights = {
    'predicted_vs_actual':
        'The model\'s price guesses line up very closely with the real car prices in almost all cases.',
    'residual_plot':
        'Most of the model\'s guesses are spot-on, with mistakes clustered very tightly around zero.',
    'feature_importance':
        'The age of the vehicle is by far the most critical factor the model uses to figure out its price.',
    'learning_curve':
        'The model is learning highly effectively, and its predictions get noticeably better as it studies more car examples.',
  };

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
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Material(
            elevation: 6,
            shadowColor: Theme.of(
              context,
            ).colorScheme.shadow.withValues(alpha: 0.2),
            borderRadius: BorderRadius.circular(16),
            color: Theme.of(context).colorScheme.surface,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: TabBar(
                controller: _tabController,
                labelColor: Theme.of(context).colorScheme.primary,
                unselectedLabelColor: Theme.of(
                  context,
                ).colorScheme.onSurfaceVariant,
                indicatorSize: TabBarIndicatorSize.tab,
                tabs: const [
                  Tab(icon: Icon(Icons.directions_car), text: 'Car Predictor'),
                  Tab(icon: Icon(Icons.home), text: 'House Predictor'),
                ],
              ),
            ),
          ),
        ),
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildMetricsCard(
                _evaluation!.carEvaluation,
                'Car Price Predictor Evaluation',
                isHouse: false,
              ),
              _buildMetricsCard(
                _evaluation!.houseEvaluation,
                'House Price Predictor Evaluation',
                isHouse: true,
              ),
            ],
          ),
        ),
      ],
    );
  }
}
