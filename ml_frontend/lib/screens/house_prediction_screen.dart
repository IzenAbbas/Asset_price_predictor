/// House Price Prediction screen — collects property features and shows
/// the predicted price from the FastAPI backend.
library;

import 'package:flutter/material.dart';
import '../models/prediction_model.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';

class HousePredictionScreen extends StatefulWidget {
  final DropdownOptions options;
  const HousePredictionScreen({super.key, required this.options});

  @override
  State<HousePredictionScreen> createState() => _HousePredictionScreenState();
}

class _HousePredictionScreenState extends State<HousePredictionScreen>
    with SingleTickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _api = ApiService();

  // Numeric controllers
  final _urlCtrl = TextEditingController();
  final _areaCtrl = TextEditingController(text: '1089');
  final _bedroomsCtrl = TextEditingController(text: '3');
  final _bathsCtrl = TextEditingController(text: '3');
  final _latCtrl = TextEditingController(text: '33.6844');
  final _lngCtrl = TextEditingController(text: '73.0479');
  final _yearCtrl = TextEditingController(text: '2022');
  final _monthCtrl = TextEditingController(text: '6');
  final _locationCtrl = TextEditingController(text: 'dha phase 6');

  // Dropdown selections
  String? _propertyType;
  String? _city;
  String? _province;
  String? _purpose;
  String _areaUnit = 'sq ft';

  // State
  bool _isLoading = false;
  bool _isExtracting = false;
  HousePredictionOutput? _result;
  String? _error;
  String? _extractionError;

  late AnimationController _animCtrl;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _fadeAnim = CurvedAnimation(parent: _animCtrl, curve: Curves.easeOut);

    final house = widget.options.house;
    _propertyType = house['property_types']?.firstOrNull;
    _city = house['cities']?.firstOrNull;
    _province = house['provinces']?.firstOrNull;
    _purpose = house['purposes']?.firstOrNull;
    _areaUnit = 'sq ft';
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    _urlCtrl.dispose();
    _areaCtrl.dispose();
    _bedroomsCtrl.dispose();
    _bathsCtrl.dispose();
    _latCtrl.dispose();
    _lngCtrl.dispose();
    _yearCtrl.dispose();
    _monthCtrl.dispose();
    _locationCtrl.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
    });
    _animCtrl.reset();

    try {
      final input = HousePredictionInput(
        totalArea: _convertToSqFt(double.parse(_areaCtrl.text), _areaUnit),
        bedrooms: int.parse(_bedroomsCtrl.text),
        baths: int.parse(_bathsCtrl.text),
        latitude: double.parse(_latCtrl.text),
        longitude: double.parse(_lngCtrl.text),
        listingYear: int.parse(_yearCtrl.text),
        listingMonth: int.parse(_monthCtrl.text),
        propertyType: _propertyType!,
        location: _locationCtrl.text.trim(),
        city: _city!,
        provinceName: _province!,
        purpose: _purpose!,
      );
      final output = await _api.predictHousePrice(input);
      setState(() {
        _result = output;
        _isLoading = false;
      });
      _animCtrl.forward();
    } catch (e) {
      setState(() {
        _error = ApiService.userMessageFromError(
          e,
          fallback:
              'We could not generate a house price right now. Please try again.',
        );
        _isLoading = false;
      });
    }
  }

  Future<void> _extractHouseFields() async {
    final url = _urlCtrl.text.trim();
    if (url.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Please enter a URL')));
      return;
    }

    setState(() {
      _isExtracting = true;
      _extractionError = null;
    });

    try {
      final extracted = await _api.extractHouseFields(url);

      if (!mounted) return;

      setState(() {
        _isExtracting = false;
        _extractionError = null;

        if (extracted.areaUnit != null && extracted.areaValue != null) {
          _areaUnit = extracted.areaUnit!;
          final area = extracted.areaValue!;
          _areaCtrl.text = area % 1 == 0
              ? area.toStringAsFixed(0)
              : area.toStringAsFixed(2);
        } else if (extracted.totalArea != null) {
          _areaUnit = 'sq ft';
          final area = extracted.totalArea!;
          _areaCtrl.text = area % 1 == 0
              ? area.toStringAsFixed(0)
              : area.toStringAsFixed(2);
        }
        if (extracted.bedrooms != null) {
          _bedroomsCtrl.text = extracted.bedrooms.toString();
        }
        if (extracted.baths != null) {
          _bathsCtrl.text = extracted.baths.toString();
        }
        if (extracted.latitude != null) {
          _latCtrl.text = extracted.latitude!.toStringAsFixed(5);
        }
        if (extracted.longitude != null) {
          _lngCtrl.text = extracted.longitude!.toStringAsFixed(5);
        }
        if (extracted.listingYear != null) {
          _yearCtrl.text = extracted.listingYear.toString();
        }
        if (extracted.listingMonth != null) {
          _monthCtrl.text = extracted.listingMonth.toString();
        }
        if (extracted.propertyType != null &&
            widget.options.house['property_types']?.contains(
                  extracted.propertyType,
                ) ==
                true) {
          _propertyType = extracted.propertyType;
        }
        if (extracted.purpose != null &&
            widget.options.house['purposes']?.contains(extracted.purpose) ==
                true) {
          _purpose = extracted.purpose;
        }
        if (extracted.city != null &&
            widget.options.house['cities']?.contains(extracted.city) == true) {
          _city = extracted.city;
        }
        if (extracted.provinceName != null &&
            widget.options.house['provinces']?.contains(
                  extracted.provinceName,
                ) ==
                true) {
          _province = extracted.provinceName;
        }
        if (extracted.location != null) {
          _locationCtrl.text = extracted.location!;
        }
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('House details extracted and populated!'),
          duration: Duration(seconds: 2),
        ),
      );
    } catch (e) {
      setState(() {
        _isExtracting = false;
        _extractionError = ApiService.userMessageFromError(
          e,
          fallback:
              'We could not read details from that listing. You can fill the form manually.',
        );
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final house = widget.options.house;
    final scheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Header ──
            Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: scheme.primary.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    Icons.home_rounded,
                    color: scheme.primary,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'House Price Predictor',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      'Estimate property value in Pakistan',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: isDark
                            ? AppColors.textSecondaryDark
                            : AppColors.textSecondaryLight,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 20),

            // ── URL Extraction Card ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(
                        Icons.link_rounded,
                        color: AppColors.primary,
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Auto-fill from Zameen',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                          color: AppColors.primary,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  TextFormField(
                    controller: _urlCtrl,
                    decoration: InputDecoration(
                      labelText: 'Listing URL',
                      hintText: 'e.g. https://www.zameen.com/...',
                      hintStyle: const TextStyle(fontSize: 12),
                    ),
                    validator: (v) {
                      if (v == null || v.isEmpty) return null;
                      if (!v.contains('http')) return 'Enter a valid URL';
                      return null;
                    },
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: _isExtracting ? null : _extractHouseFields,
                      icon: _isExtracting
                          ? const SizedBox(
                              height: 18,
                              width: 18,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  AppColors.accent,
                                ),
                              ),
                            )
                          : const Icon(Icons.auto_fix_high),
                      label: Text(
                        _isExtracting ? 'Extracting...' : 'Extract Details',
                      ),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        foregroundColor: AppColors.primary,
                        side: BorderSide(
                          color: AppColors.primary.withValues(alpha: 0.45),
                        ),
                      ),
                    ),
                  ),
                  if (_extractionError != null) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: AppColors.error.withValues(alpha: 0.08),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: AppColors.error.withValues(alpha: 0.18),
                        ),
                      ),
                      child: Row(
                        children: [
                          const Icon(
                            Icons.error_outline,
                            color: AppColors.error,
                            size: 18,
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              _extractionError!,
                              style: Theme.of(context).textTheme.bodySmall
                                  ?.copyWith(color: AppColors.error),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── Property Details ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Property Details'),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: _numField(_areaCtrl, 'Total Area', 'e.g. 7'),
                      ),
                      const SizedBox(width: 12),
                      SizedBox(width: 130, child: _areaUnitDropdown()),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                        child: _numField(
                          _bedroomsCtrl,
                          'Bedrooms',
                          'e.g. 3',
                          isInt: true,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _numField(
                          _bathsCtrl,
                          'Baths',
                          'e.g. 3',
                          isInt: true,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  _textField(_locationCtrl, 'Location', 'e.g. dha phase 6'),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // ── Geography ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Geography'),
                  const SizedBox(height: 16),
                  _dropdown(
                    'City',
                    house['cities'] ?? [],
                    _city,
                    (v) => setState(() => _city = v),
                  ),
                  const SizedBox(height: 14),
                  _dropdown(
                    'Province',
                    house['provinces'] ?? [],
                    _province,
                    (v) => setState(() => _province = v),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // ── Listing Info ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Listing Info'),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: _numField(
                          _yearCtrl,
                          'Listing Year',
                          'e.g. 2022',
                          isInt: true,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _numField(
                          _monthCtrl,
                          'Month',
                          'e.g. 6',
                          isInt: true,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  _dropdown(
                    'Property Type',
                    house['property_types'] ?? [],
                    _propertyType,
                    (v) => setState(() => _propertyType = v),
                  ),
                  const SizedBox(height: 14),
                  _dropdown(
                    'Purpose',
                    house['purposes'] ?? [],
                    _purpose,
                    (v) => setState(() => _purpose = v),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // ── Submit ──
            _predictButton(),
            const SizedBox(height: 20),

            if (_result != null) _resultCard(),
            if (_error != null) _errorCard(),

            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  // ─── Helpers ───────────────────────────────────────────────────────

  Widget _sectionTitle(String title) {
    return Text(
      title,
      style: Theme.of(context).textTheme.titleSmall?.copyWith(
        fontWeight: FontWeight.w600,
        color: Theme.of(context).colorScheme.primary,
      ),
    );
  }

  double _convertToSqFt(double value, String unit) {
    switch (unit) {
      case 'marla':
        return value * 272.25;
      case 'kanal':
        return value * 20 * 272.25;
      case 'sq yd':
        return value * 9;
      case 'sq m':
        return value * 10.7639;
      case 'acre':
        return value * 43560;
      default:
        return value;
    }
  }

  Widget _numField(
    TextEditingController ctrl,
    String label,
    String hint, {
    bool isInt = false,
  }) {
    return TextFormField(
      controller: ctrl,
      keyboardType: isInt
          ? TextInputType.number
          : const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(labelText: label, hintText: hint),
      validator: (v) {
        if (v == null || v.isEmpty) return 'Required';
        if (isInt && int.tryParse(v) == null) return 'Valid integer needed';
        if (!isInt && double.tryParse(v) == null) return 'Valid number needed';
        return null;
      },
    );
  }

  Widget _textField(TextEditingController ctrl, String label, String hint) {
    return TextFormField(
      controller: ctrl,
      decoration: InputDecoration(labelText: label, hintText: hint),
      validator: (v) => (v == null || v.isEmpty) ? 'Required' : null,
    );
  }

  Widget _dropdown(
    String label,
    List<String> items,
    String? value,
    ValueChanged<String?> onChanged,
  ) {
    return DropdownButtonFormField<String>(
      initialValue: items.contains(value) ? value : null,
      isExpanded: true,
      dropdownColor: Theme.of(context).colorScheme.surface,
      decoration: InputDecoration(labelText: label),
      items: items
          .map((e) => DropdownMenuItem(value: e, child: Text(e)))
          .toList(),
      onChanged: onChanged,
      validator: (v) => v == null ? 'Please select' : null,
    );
  }

  Widget _areaUnitDropdown() {
    const units = ['sq ft', 'marla', 'kanal', 'sq yd', 'sq m', 'acre'];
    return DropdownButtonFormField<String>(
      value: units.contains(_areaUnit) ? _areaUnit : 'sq ft',
      isExpanded: true,
      dropdownColor: Theme.of(context).colorScheme.surface,
      decoration: const InputDecoration(labelText: 'Unit'),
      items: units
          .map((e) => DropdownMenuItem(value: e, child: Text(e)))
          .toList(),
      onChanged: (v) => setState(() => _areaUnit = v ?? 'sq ft'),
    );
  }

  Widget _predictButton() {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        onPressed: _isLoading ? null : _predict,
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        icon: _isLoading
            ? const SizedBox(
                height: 18,
                width: 18,
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 2.2,
                ),
              )
            : const Icon(Icons.auto_awesome, size: 20),
        label: Text(_isLoading ? 'Predicting...' : 'Predict House Price'),
      ),
    );
  }

  Widget _resultCard() {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return FadeTransition(
      opacity: _fadeAnim,
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: isDark ? 0.2 : 0.04),
              blurRadius: 18,
              offset: const Offset(0, 8),
            ),
          ],
          border: Border.all(
            color: isDark ? AppColors.borderDark : AppColors.borderLightTheme,
          ),
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Icon(
              Icons.check_circle_rounded,
              color: Theme.of(context).colorScheme.primary,
              size: 42,
            ),
            const SizedBox(height: 12),
            Text(
              'Predicted Price',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: isDark
                    ? AppColors.textSecondaryDark
                    : AppColors.textSecondaryLight,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              _result!.formattedPrice,
              style: Theme.of(context).textTheme.displaySmall?.copyWith(
                fontWeight: FontWeight.w800,
                color: isDark
                    ? AppColors.textPrimaryDark
                    : AppColors.textPrimaryLight,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _errorCard() {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.error.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.error.withValues(alpha: 0.18)),
      ),
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          const Icon(
            Icons.info_outline_rounded,
            color: AppColors.error,
            size: 24,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _error!,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: AppColors.error),
            ),
          ),
        ],
      ),
    );
  }
}
