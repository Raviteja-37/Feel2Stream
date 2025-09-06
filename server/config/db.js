// config/db.js
const mongoose = require('mongoose');

module.exports = function connectDB() {
  const uri = process.env.MONGO_URI;
  if (!uri) {
    console.error('MONGO_URI not set in .env');
    process.exit(1);
  }
  mongoose.set('strictQuery', true);
  mongoose
    .connect(uri)
    .then(() => console.log('✅ MongoDB connected'))
    .catch((err) => {
      console.error('❌ MongoDB connection error', err.message || err);
      process.exit(1);
    });
};
