import 'package:dio/dio.dart';

final dio = Dio();

final String host = "http://localhost:5000";

Future<String> fetchAnalysis(String stockCode) async {
  final response = await dio.get('$host/analyze/$stockCode');
  return response.data;
  // return getNumberLine(30);
}

String getNumberLine(int i) {
  String s = "";
  for (int j = 0; j < i; j++) {
    s += "$j\n";
  }
  return s;
}
