/// Car Price Prediction screen — collects car features and shows the
/// predicted price from the FastAPI backend.
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/prediction_model.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';

class CarPredictionScreen extends StatefulWidget {
  final DropdownOptions options;
  const CarPredictionScreen({super.key, required this.options});

  @override
  State<CarPredictionScreen> createState() => _CarPredictionScreenState();
}

class _CarPredictionScreenState extends State<CarPredictionScreen>
    with SingleTickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _api = ApiService();

  // Controllers
  final _modelYearCtrl = TextEditingController(text: '2020');
  final _mileageCtrl = TextEditingController(text: '50000');
  final _engineCapCtrl = TextEditingController(text: '1300');

  // Dropdown selections
  String? _fuelType;
  String? _transmission;
  String? _assembly;
  String? _brand;
  String? _modelName;

  // State
  bool _isLoading = false;
  CarPredictionOutput? _result;
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

    // Set default dropdown values
    final car = widget.options.car;
    _fuelType = car['fuel_types']?.firstOrNull;
    _transmission = car['transmissions']?.firstOrNull;
    _assembly = car['assemblies']?.firstOrNull;
    _brand = car['brands']?.firstOrNull;
    _modelName = car['model_names']?.firstOrNull;
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    _modelYearCtrl.dispose();
    _mileageCtrl.dispose();
    _engineCapCtrl.dispose();
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
      final input = CarPredictionInput(
        modelYear: int.parse(_modelYearCtrl.text),
        mileage: double.parse(_mileageCtrl.text),
        engineCapacity: double.parse(_engineCapCtrl.text),
        fuelType: _fuelType!,
        transmission: _transmission!,
        assembly: _assembly!,
        brand: _brand!,
        modelName: _modelName!,
      );
      final output = await _api.predictCarPrice(input);
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
    final car = widget.options.car;

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
                    gradient: AppGradients.primary,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: const Icon(Icons.directions_car_rounded,
                      color: Colors.white, size: 28),
                ),
                const SizedBox(width: 14),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Car Price Predictor',
                        style: GoogleFonts.inter(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary,
                        )),
                    Text('Estimate used car value in Pakistan',
                        style: GoogleFonts.inter(
                          fontSize: 13,
                          color: AppColors.textSecondary,
                        )),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 24),

            // ── Numeric Fields ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Vehicle Details'),
                  const SizedBox(height: 16),
                  _numField(_modelYearCtrl, 'Model Year', 'e.g. 2020',
                      isInt: true),
                  const SizedBox(height: 14),
                  _numField(_mileageCtrl, 'Mileage (km)', 'e.g. 50000'),
                  const SizedBox(height: 14),
                  _numField(_engineCapCtrl, 'Engine Capacity (cc)', 'e.g. 1300'),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // ── Dropdown Fields ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Specifications'),
                  const SizedBox(height: 16),
                  _dropdown('Fuel Type', car['fuel_types'] ?? [], _fuelType,
                      (v) => setState(() => _fuelType = v)),
                  const SizedBox(height: 14),
                  _dropdown('Transmission', car['transmissions'] ?? [],
                      _transmission, (v) => setState(() => _transmission = v)),
                  const SizedBox(height: 14),
                  _dropdown('Assembly', car['assemblies'] ?? [], _assembly,
                      (v) => setState(() => _assembly = v)),
                  const SizedBox(height: 14),
                  _dropdown('Brand', car['brands'] ?? [], _brand,
                      (v) => setState(() => _brand = v)),
                  const SizedBox(height: 14),
                  _dropdown('Model Name', car['model_names'] ?? [], _modelName,
                      (v) => setState(() => _modelName = v)),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // ── Submit Button ──
            _predictButton(),
            const SizedBox(height: 20),

            // ── Result / Error ──
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
        if (isInt && int.tryParse(v) == null) return 'Enter a valid integer';
        if (!isInt && double.tryParse(v) == null) return 'Enter a valid number';
        return null;
      },
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
        gradient: AppGradients.primary,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withValues(alpha: 0.4),
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
                child:
                    CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5))
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.auto_awesome, size: 20),
                  const SizedBox(width: 10),
                  Text('Predict Car Price',
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
