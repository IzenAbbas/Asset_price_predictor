/// House Price Prediction screen — collects property features and shows
/// the predicted price from the FastAPI backend.
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
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

  // State
  bool _isLoading = false;
  HousePredictionOutput? _result;
  String? _error;

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
  }

  @override
  void dispose() {
    _animCtrl.dispose();
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
        totalArea: double.parse(_areaCtrl.text),
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
        _error = e.toString().replaceFirst('Exception: ', '');
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final house = widget.options.house;

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Header ──
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    gradient: AppGradients.accent,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: const Icon(Icons.home_rounded,
                      color: Colors.white, size: 28),
                ),
                const SizedBox(width: 14),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('House Price Predictor',
                        style: GoogleFonts.inter(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary,
                        )),
                    Text('Estimate property value in Pakistan',
                        style: GoogleFonts.inter(
                          fontSize: 13,
                          color: AppColors.textSecondary,
                        )),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 24),

            // ── Property Details ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Property Details'),
                  const SizedBox(height: 16),
                  _numField(_areaCtrl, 'Total Area (sq ft)', 'e.g. 1089'),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                          child: _numField(
                              _bedroomsCtrl, 'Bedrooms', 'e.g. 3',
                              isInt: true)),
                      const SizedBox(width: 12),
                      Expanded(
                          child: _numField(_bathsCtrl, 'Baths', 'e.g. 3',
                              isInt: true)),
                    ],
                  ),
                  const SizedBox(height: 14),
                  _textField(
                      _locationCtrl, 'Location', 'e.g. dha phase 6'),
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
                  Row(
                    children: [
                      Expanded(
                          child: _numField(_latCtrl, 'Latitude', 'e.g. 33.68')),
                      const SizedBox(width: 12),
                      Expanded(
                          child: _numField(
                              _lngCtrl, 'Longitude', 'e.g. 73.05')),
                    ],
                  ),
                  const SizedBox(height: 14),
                  _dropdown('City', house['cities'] ?? [], _city,
                      (v) => setState(() => _city = v)),
                  const SizedBox(height: 14),
                  _dropdown('Province', house['provinces'] ?? [], _province,
                      (v) => setState(() => _province = v)),
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
                              _yearCtrl, 'Listing Year', 'e.g. 2022',
                              isInt: true)),
                      const SizedBox(width: 12),
                      Expanded(
                          child: _numField(
                              _monthCtrl, 'Month', 'e.g. 6',
                              isInt: true)),
                    ],
                  ),
                  const SizedBox(height: 14),
                  _dropdown(
                      'Property Type',
                      house['property_types'] ?? [],
                      _propertyType,
                      (v) => setState(() => _propertyType = v)),
                  const SizedBox(height: 14),
                  _dropdown('Purpose', house['purposes'] ?? [], _purpose,
                      (v) => setState(() => _purpose = v)),
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
    return Text(title,
        style: GoogleFonts.inter(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            color: AppColors.accent));
  }

  Widget _numField(TextEditingController ctrl, String label, String hint,
      {bool isInt = false}) {
    return TextFormField(
      controller: ctrl,
      keyboardType: isInt
          ? TextInputType.number
          : const TextInputType.numberWithOptions(decimal: true),
      style: const TextStyle(color: AppColors.textPrimary),
      decoration: InputDecoration(labelText: label, hintText: hint),
      validator: (v) {
        if (v == null || v.isEmpty) return 'Required';
        if (isInt && int.tryParse(v) == null) return 'Valid integer needed';
        if (!isInt && double.tryParse(v) == null) return 'Valid number needed';
        return null;
      },
    );
  }

  Widget _textField(
      TextEditingController ctrl, String label, String hint) {
    return TextFormField(
      controller: ctrl,
      style: const TextStyle(color: AppColors.textPrimary),
      decoration: InputDecoration(labelText: label, hintText: hint),
      validator: (v) => (v == null || v.isEmpty) ? 'Required' : null,
    );
  }

  Widget _dropdown(String label, List<String> items, String? value,
      ValueChanged<String?> onChanged) {
    return DropdownButtonFormField<String>(
      initialValue: items.contains(value) ? value : null,
      isExpanded: true,
      dropdownColor: AppColors.bgCard,
      style: const TextStyle(color: AppColors.textPrimary),
      decoration: InputDecoration(labelText: label),
      items: items
          .map((e) => DropdownMenuItem(value: e, child: Text(e)))
          .toList(),
      onChanged: onChanged,
      validator: (v) => v == null ? 'Please select' : null,
    );
  }

  Widget _predictButton() {
    return Container(
      decoration: BoxDecoration(
        gradient: AppGradients.accent,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: AppColors.accent.withValues(alpha: 0.4),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: ElevatedButton(
        onPressed: _isLoading ? null : _predict,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          padding: const EdgeInsets.symmetric(vertical: 18),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
        child: _isLoading
            ? const SizedBox(
                height: 22,
                width: 22,
                child: CircularProgressIndicator(
                    color: Colors.white, strokeWidth: 2.5))
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.auto_awesome, size: 20),
                  const SizedBox(width: 10),
                  Text('Predict House Price',
                      style: GoogleFonts.inter(
                          fontSize: 16, fontWeight: FontWeight.w600)),
                ],
              ),
      ),
    );
  }

  Widget _resultCard() {
    return FadeTransition(
      opacity: _fadeAnim,
      child: Container(
        decoration: BoxDecoration(
          gradient: AppGradients.success,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: AppColors.success.withValues(alpha: 0.3),
              blurRadius: 20,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const Icon(Icons.check_circle_rounded,
                color: Colors.white, size: 48),
            const SizedBox(height: 12),
            Text('Predicted Price',
                style: GoogleFonts.inter(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                    color: Colors.white70)),
            const SizedBox(height: 6),
            Text(_result!.formattedPrice,
                style: GoogleFonts.inter(
                    fontSize: 32,
                    fontWeight: FontWeight.w800,
                    color: Colors.white)),
          ],
        ),
      ),
    );
  }

  Widget _errorCard() {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.error.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.error.withValues(alpha: 0.4)),
      ),
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: AppColors.error, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Text(_error!,
                style: GoogleFonts.inter(
                    fontSize: 13, color: AppColors.error)),
          ),
        ],
      ),
    );
  }
}
