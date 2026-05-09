/// Entry point for the Asset Advisor Flutter app.
///
/// Fetches dropdown options from the FastAPI backend on startup, then
/// shows a bottom-navigation shell with Car and House prediction tabs.
library;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'models/prediction_model.dart';
import 'screens/car_prediction_screen.dart';
import 'screens/house_prediction_screen.dart';
import 'services/api_service.dart';
import 'theme/app_theme.dart';

void main() {
  runApp(const AssetAdvisorApp());
}

class AssetAdvisorApp extends StatelessWidget {
  const AssetAdvisorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Asset Advisor',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: const AppShell(),
    );
  }
}

/// Top-level shell that loads dropdown options once, then renders
/// car/house tabs with a smooth bottom navigation bar.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  final _api = ApiService();
  int _currentTab = 0;

  DropdownOptions? _options;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadOptions();
  }

  Future<void> _loadOptions() async {
    try {
      final options = await _api.getDropdownOptions();
      setState(() {
        _options = options;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString().replaceFirst('Exception: ', '');
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // ── App Bar ──
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            ShaderMask(
              shaderCallback: (bounds) =>
                  AppGradients.primary.createShader(bounds),
              child: const Icon(
                Icons.analytics_rounded,
                color: Colors.white,
                size: 26,
              ),
            ),
            const SizedBox(width: 10),
            const GradientText(
              'Asset Advisor',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
            ),
          ],
        ),
      ),

      // ── Body ──
      body: _buildBody(),

      // ── Bottom Navigation ──
      bottomNavigationBar: _options != null
          ? Container(
              decoration: BoxDecoration(
                color: AppColors.bgCard,
                border: const Border(top: BorderSide(color: AppColors.border)),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.3),
                    blurRadius: 16,
                    offset: const Offset(0, -4),
                  ),
                ],
              ),
              child: NavigationBar(
                backgroundColor: Colors.transparent,
                indicatorColor: AppColors.primary.withValues(alpha: 0.2),
                selectedIndex: _currentTab,
                onDestinationSelected: (i) => setState(() => _currentTab = i),
                destinations: const [
                  NavigationDestination(
                    icon: Icon(
                      Icons.directions_car_outlined,
                      color: AppColors.textSecondary,
                    ),
                    selectedIcon: Icon(
                      Icons.directions_car_rounded,
                      color: AppColors.primaryLight,
                    ),
                    label: 'Car',
                  ),
                  NavigationDestination(
                    icon: Icon(
                      Icons.home_outlined,
                      color: AppColors.textSecondary,
                    ),
                    selectedIcon: Icon(
                      Icons.home_rounded,
                      color: AppColors.accent,
                    ),
                    label: 'House',
                  ),
                ],
              ),
            )
          : null,
    );
  }

  Widget _buildBody() {
    // Loading state
    if (_loading) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(color: AppColors.primaryLight),
            const SizedBox(height: 20),
            Text(
              'Connecting to prediction server...',
              style: GoogleFonts.inter(
                fontSize: 14,
                color: AppColors.textSecondary,
              ),
            ),
          ],
        ),
      );
    }

    // Error state
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: GlassCard(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.cloud_off_rounded,
                  color: AppColors.error,
                  size: 56,
                ),
                const SizedBox(height: 16),
                Text(
                  'Connection Error',
                  style: GoogleFonts.inter(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: GoogleFonts.inter(
                    fontSize: 13,
                    color: AppColors.textSecondary,
                  ),
                ),
                const SizedBox(height: 20),
                ElevatedButton.icon(
                  onPressed: () {
                    setState(() {
                      _loading = true;
                      _error = null;
                    });
                    _loadOptions();
                  },
                  icon: const Icon(Icons.refresh),
                  label: const Text('Retry'),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // Tabbed content
    return IndexedStack(
      index: _currentTab,
      children: [
        CarPredictionScreen(options: _options!),
        HousePredictionScreen(options: _options!),
      ],
    );
  }
}
