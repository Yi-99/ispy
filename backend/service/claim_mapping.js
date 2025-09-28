// === Common normalizers (use across carriers) ===
function binVehiclePrice(v) {
    if (v == null) return "unknown";
    const x = Number(v);
    if (!isFinite(x)) return String(v); // 이미 구간 문자열이면 그대로
    if (x < 11000) return "less_than_20000";
    if (x < 20000) return "11000_to_20000";
    if (x < 29000) return "20000_to_29000";
    if (x < 39000) return "30000_to_39000";
    if (x < 52000) return "39000_to_52000";
    if (x < 69000) return "52000_to_69000";
    return "more_than_69000";
  }
  const yesNo = (v) => (String(v).toLowerCase().startsWith('y') ? 'Yes' : 'No');
  const yn = (s) => ({'y':'Yes','n':'No'}[String(s).toLowerCase()] || String(s));
  const dowMap = {mon:'Monday',tue:'Tuesday',wed:'Wednesday',thu:'Thursday',fri:'Friday',sat:'Saturday',sun:'Sunday'};
  const monMap = {jan:'Jan',feb:'Feb',mar:'Mar',apr:'Apr',may:'May',jun:'Jun',jul:'Jul',aug:'Aug',sep:'Sep',oct:'Oct',nov:'Nov',dec:'Dec'};
  const normDow = (s) => dowMap[String(s).slice(0,3).toLowerCase()] || String(s);
  const normMon = (s) => monMap[String(s).slice(0,3).toLowerCase()] || String(s);
  function stripLeak(o){ delete o.MonthClaimed; delete o.DayOfWeekClaimed; delete o.WeekOfMonthClaimed; return o; }

  
  function mapGeico(src){
    const s = stripLeak({...src});
    return {
      "Month": normMon(s.month || s.incident_month),
      "WeekOfMonth": Number(s.week_of_month ?? 1),
      "DayOfWeek": normDow(s.day_of_week || s.incident_dow),
      "Make": s.vehicle_make || s.make,
      "AccidentArea": (s.location_type==='urban'||s.accident_area==='Urban') ? 'Urban' : 'Rural',
      "Sex": s.driver_sex || s.sex,
      "MaritalStatus": s.marital_status || 'Single',
      "Age": Number(s.driver_age ?? s.age),
      "Fault": s.fault==='third_party' ? 'Third Party':'Policy Holder',
      "PolicyType": s.policy_type || 'Sedan',
      "VehicleCategory": s.vehicle_category || 'Sedan',
      "VehiclePrice": binVehiclePrice(s.vehicle_price),
      "PolicyNumber": Number(s.policy_number),
      "RepNumber": Number(s.rep_number ?? 1),
      "Deductible": Number(s.deductible ?? 500),
      "DriverRating": Number(s.driver_rating ?? 3),
      "Days:Policy-Accident": Number(s.days_policy_accident ?? s.days_since_policy_to_accident),
      "Days:Policy-Claim": Number(s.days_policy_claim ?? s.days_since_policy_to_claim),
      "PastNumberOfClaims": Number(s.past_claims ?? 0),
      "AgeOfVehicle": Number(s.vehicle_age ?? s.age_of_vehicle ?? 0),
      "AgeOfPolicyHolder": Number(s.policy_holder_age ?? s.age_of_policy_holder ?? s.driver_age),
      "PoliceReportFiled": yesNo(s.police_report_filed ?? s.police_report),
      "WitnessPresent": yesNo(s.witness_present ?? s.witness),
      "AgentType": s.agent_type ? (s.agent_type==='internal'?'Internal':'External') : 'External',
      "NumberOfSuppliments": Number(s.num_suppliments ?? 0),
      "AddressChange-Claim": s.address_change_claim ?? 'no change',
      "NumberOfCars": Number(s.num_cars ?? 1),
      "Year": Number(s.vehicle_year ?? s.year),
      "BasePolicy": s.base_policy || 'Liability',
      "ClaimAmount": Number(s.claim_amount ?? 0)
    };
  }

  function mapStateFarm(src){
    const s = stripLeak({...src});
    return {
      "Month": normMon(s.month),
      "WeekOfMonth": Number(s.weekOfMonth ?? 1),
      "DayOfWeek": normDow(s.dayOfWeek),
      "Make": s.make,
      "AccidentArea": s.accidentArea || (s.location==='Urban'?'Urban':'Rural'),
      "Sex": s.sex,
      "MaritalStatus": s.maritalStatus,
      "Age": Number(s.age),
      "Fault": s.fault,
      "PolicyType": s.policyType,
      "VehicleCategory": s.vehicleCategory,
      "VehiclePrice": typeof s.vehiclePrice==='number' ? binVehiclePrice(s.vehiclePrice) : s.vehiclePrice,
      "PolicyNumber": Number(s.policyNumber),
      "RepNumber": Number(s.repNumber ?? 1),
      "Deductible": Number(s.deductible ?? 500),
      "DriverRating": Number(s.driverRating ?? 3),
      "Days:Policy-Accident": Number(s.daysPolicyAccident),
      "Days:Policy-Claim": Number(s.daysPolicyClaim),
      "PastNumberOfClaims": Number(s.pastNumberOfClaims ?? 0),
      "AgeOfVehicle": Number(s.ageOfVehicle ?? 0),
      "AgeOfPolicyHolder": Number(s.ageOfPolicyHolder ?? s.age),
      "PoliceReportFiled": s.policeReportFiled ?? 'No',
      "WitnessPresent": s.witnessPresent ?? 'No',
      "AgentType": s.agentType ?? 'Internal',
      "NumberOfSuppliments": Number(s.numberOfSuppliments ?? 0),
      "AddressChange-Claim": s.addressChangeClaim ?? 'no change',
      "NumberOfCars": Number(s.numberOfCars ?? 1),
      "Year": Number(s.year),
      "BasePolicy": s.basePolicy,
      "ClaimAmount": Number(s.claimAmount ?? 0)
    };
  }

  function mapProgressive(src){
    const s = stripLeak({...src});
    const sexMap = (v)=>({'M':'Male','F':'Female'}[String(v).toUpperCase()]||String(v));
    return {
      "Month": normMon(s.month || s.incidentMonth),
      "WeekOfMonth": Number(s["week-of-month"] ?? s.weekOfMonth ?? 1),
      "DayOfWeek": normDow(s["day-of-week"] || s.dayOfWeek),
      "Make": s.make,
      "AccidentArea": s["accident-area"] || s.accidentArea || 'Urban',
      "Sex": sexMap(s.sex ?? 'M'),
      "MaritalStatus": s.maritalStatus || 'Single',
      "Age": Number(s.age),
      "Fault": s.fault || 'Policy Holder',
      "PolicyType": s.policyType || 'Sedan',
      "VehicleCategory": s.vehicleCategory || 'Sedan',
      "VehiclePrice": binVehiclePrice(s["vehicle-price"] ?? s.vehiclePrice),
      "PolicyNumber": Number(s.policyNumber ?? s["policy-number"]),
      "RepNumber": Number(s.repNumber ?? s["rep-number"] ?? 1),
      "Deductible": Number(s.deductible ?? 500),
      "DriverRating": Number(s.driverRating ?? s["driver-rating"] ?? 3),
      "Days:Policy-Accident": Number(s["days-policy-accident"] ?? s.daysPolicyAccident),
      "Days:Policy-Claim": Number(s["days-policy-claim"] ?? s.daysPolicyClaim),
      "PastNumberOfClaims": Number(s["past-claims"] ?? s.pastNumberOfClaims ?? 0),
      "AgeOfVehicle": Number(s["vehicle-age"] ?? s.ageOfVehicle ?? 0),
      "AgeOfPolicyHolder": Number(s["policyholder-age"] ?? s.ageOfPolicyHolder ?? s.age),
      "PoliceReportFiled": yn(s["police-report"] ?? s.policeReportFiled ?? 'N'),
      "WitnessPresent": yn(s["witness"] ?? s.witnessPresent ?? 'N'),
      "AgentType": s["agent-type"] || s.agentType || 'External',
      "NumberOfSuppliments": Number(s["num-suppliments"] ?? s.numberOfSuppliments ?? 0),
      "AddressChange-Claim": s["address-change-claim"] ?? s.addressChangeClaim ?? 'no change',
      "NumberOfCars": Number(s["num-cars"] ?? s.numberOfCars ?? 1),
      "Year": Number(s.year),
      "BasePolicy": s["base-policy"] ?? s.basePolicy ?? 'Liability',
      "ClaimAmount": Number(s["claim-amount"] ?? s.claimAmount ?? 0)
    };
  }

  function mapAllstate(src){
    const s = stripLeak({...src});
    return {
      "Month": normMon(s.month),
      "WeekOfMonth": Number(s.weekOfMonth ?? 1),
      "DayOfWeek": normDow(s.dayOfWeek),
      "Make": s.make,
      "AccidentArea": s.accidentArea || (s.locationType==='Urban'?'Urban':'Rural'),
      "Sex": s.sex,
      "MaritalStatus": s.maritalStatus,
      "Age": Number(s.age),
      "Fault": s.fault,
      "PolicyType": s.policyType,
      "VehicleCategory": s.vehicleCategory,
      "VehiclePrice": typeof s.vehiclePrice==='number' ? binVehiclePrice(s.vehiclePrice) : s.vehiclePrice,
      "PolicyNumber": Number(s.policyNumber),
      "RepNumber": Number(s.repNumber ?? 1),
      "Deductible": Number(s.deductible ?? 500),
      "DriverRating": Number(s.driverRating ?? 3),
      "Days:Policy-Accident": Number(s.daysSincePolicyToAccident ?? s.daysPolicyAccident),
      "Days:Policy-Claim": Number(s.daysSincePolicyToClaim ?? s.daysPolicyClaim),
      "PastNumberOfClaims": Number(s.pastNumberOfClaims ?? 0),
      "AgeOfVehicle": Number(s.ageOfVehicle ?? 0),
      "AgeOfPolicyHolder": Number(s.ageOfPolicyHolder ?? s.age),
      "PoliceReportFiled": s.policeReportFiled ?? 'No',
      "WitnessPresent": s.witnessPresent ?? 'No',
      "AgentType": s.agentType ?? 'Internal',
      "NumberOfSuppliments": Number(s.numberOfSuppliments ?? 0),
      "AddressChange-Claim": s.addressChangeClaim ?? 'no change',
      "NumberOfCars": Number(s.numberOfCars ?? 1),
      "Year": Number(s.year),
      "BasePolicy": s.basePolicy ?? 'Liability',
      "ClaimAmount": Number(s.claimAmount ?? 0)
    };
  }

  // Export functions for use in other modules
  module.exports = {
    mapGeico,
    mapStateFarm,
    mapProgressive,
    mapAllstate,
    binVehiclePrice,
    stripLeak
  };
