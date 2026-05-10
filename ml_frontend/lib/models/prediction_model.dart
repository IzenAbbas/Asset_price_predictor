/// Data models matching the FastAPI Pydantic schemas.
///
/// These mirror [schemas.py] on the backend so that JSON serialization
/// is consistent between client and server.
library;

// ── Car Prediction ──────────────────────────────────────────────────

class CarPredictionInput {
  final int modelYear;
  final double mileage;
  final double engineCapacity;
  final String fuelType;
  final String transmission;
  final String assembly;
  final String brand;
  final String modelName;

  CarPredictionInput({
    required this.modelYear,
    required this.mileage,
    required this.engineCapacity,
    required this.fuelType,
    required this.transmission,
    required this.assembly,
    required this.brand,
    required this.modelName,
  });

  Map<String, dynamic> toJson() => {
    'model_year': modelYear,
    'mileage': mileage,
    'engine_capacity': engineCapacity,
    'fuel_type': fuelType,
    'transmission': transmission,
    'assembly': assembly,
    'brand': brand,
    'model_name': modelName,
  };
}

class CarPredictionOutput {
  final double predictedPrice;
  final String formattedPrice;

  CarPredictionOutput({
    required this.predictedPrice,
    required this.formattedPrice,
  });

  factory CarPredictionOutput.fromJson(Map<String, dynamic> json) {
    return CarPredictionOutput(
      predictedPrice: (json['predicted_price'] as num).toDouble(),
      formattedPrice: json['formatted_price'] as String,
    );
  }
}

// ── Vehicle Field Extraction (NER) ──────────────────────────────────

class VehicleFieldsOutput {
  final int? modelYear;
  final double? mileageKm;
  final double? engineCapacityCc;
  final String? fuelType;
  final String? transmission;
  final String? assembly;
  final String? brand;
  final String? modelName;

  VehicleFieldsOutput({
    this.modelYear,
    this.mileageKm,
    this.engineCapacityCc,
    this.fuelType,
    this.transmission,
    this.assembly,
    this.brand,
    this.modelName,
  });

  factory VehicleFieldsOutput.fromJson(Map<String, dynamic> json) {
    return VehicleFieldsOutput(
      modelYear: json['model_year'] as int?,
      mileageKm: (json['mileage_km'] as num?)?.toDouble(),
      engineCapacityCc: (json['engine_capacity_cc'] as num?)?.toDouble(),
      fuelType: json['fuel_type'] as String?,
      transmission: json['transmission'] as String?,
      assembly: json['assembly'] as String?,
      brand: json['brand'] as String?,
      modelName: json['model_name'] as String?,
    );
  }
}

// ── House Field Extraction (NER) ──────────────────────────────────

class HouseFieldsOutput {
  final double? totalArea;
  final int? bedrooms;
  final int? baths;
  final double? latitude;
  final double? longitude;
  final int? listingYear;
  final int? listingMonth;
  final String? propertyType;
  final String? location;
  final String? city;
  final String? provinceName;
  final String? purpose;

  HouseFieldsOutput({
    this.totalArea,
    this.bedrooms,
    this.baths,
    this.latitude,
    this.longitude,
    this.listingYear,
    this.listingMonth,
    this.propertyType,
    this.location,
    this.city,
    this.provinceName,
    this.purpose,
  });

  factory HouseFieldsOutput.fromJson(Map<String, dynamic> json) {
    return HouseFieldsOutput(
      totalArea: (json['Total_Area'] as num?)?.toDouble(),
      bedrooms: json['bedrooms'] as int?,
      baths: json['baths'] as int?,
      latitude: (json['latitude'] as num?)?.toDouble(),
      longitude: (json['longitude'] as num?)?.toDouble(),
      listingYear: json['listing_year'] as int?,
      listingMonth: json['listing_month'] as int?,
      propertyType: json['property_type'] as String?,
      location: json['location'] as String?,
      city: json['city'] as String?,
      provinceName: json['province_name'] as String?,
      purpose: json['purpose'] as String?,
    );
  }
}

// ── House Prediction ────────────────────────────────────────────────

class HousePredictionInput {
  final double totalArea;
  final int bedrooms;
  final int baths;
  final double latitude;
  final double longitude;
  final int listingYear;
  final int listingMonth;
  final String propertyType;
  final String location;
  final String city;
  final String provinceName;
  final String purpose;

  HousePredictionInput({
    required this.totalArea,
    required this.bedrooms,
    required this.baths,
    required this.latitude,
    required this.longitude,
    required this.listingYear,
    required this.listingMonth,
    required this.propertyType,
    required this.location,
    required this.city,
    required this.provinceName,
    required this.purpose,
  });

  Map<String, dynamic> toJson() => {
    'Total_Area': totalArea,
    'bedrooms': bedrooms,
    'baths': baths,
    'latitude': latitude,
    'longitude': longitude,
    'listing_year': listingYear,
    'listing_month': listingMonth,
    'property_type': propertyType,
    'location': location,
    'city': city,
    'province_name': provinceName,
    'purpose': purpose,
  };
}

class HousePredictionOutput {
  final double predictedPrice;
  final String formattedPrice;

  HousePredictionOutput({
    required this.predictedPrice,
    required this.formattedPrice,
  });

  factory HousePredictionOutput.fromJson(Map<String, dynamic> json) {
    return HousePredictionOutput(
      predictedPrice: (json['predicted_price'] as num).toDouble(),
      formattedPrice: json['formatted_price'] as String,
    );
  }
}

// ── Dropdown Options ────────────────────────────────────────────────

class DropdownOptions {
  final Map<String, List<String>> car;
  final Map<String, List<String>> house;
  final Map<String, List<String>> carBrandModels;

  DropdownOptions({
    required this.car,
    required this.house,
    required this.carBrandModels,
  });

  factory DropdownOptions.fromJson(Map<String, dynamic> json) {
    return DropdownOptions(
      car: _parseOptionMap(json['car'] as Map<String, dynamic>),
      house: _parseOptionMap(json['house'] as Map<String, dynamic>),
      carBrandModels: json.containsKey('car_brand_models')
          ? _parseOptionMap(json['car_brand_models'] as Map<String, dynamic>)
          : {},
    );
  }

  static Map<String, List<String>> _parseOptionMap(Map<String, dynamic> map) {
    return map.map(
      (key, value) =>
          MapEntry(key, (value as List<dynamic>).cast<String>().toList()),
    );
  }
}
