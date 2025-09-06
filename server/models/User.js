// models/User.js
const { Schema, model } = require('mongoose');

const RecommendationSchema = new Schema(
  {
    emotion: { type: String, required: true },
    items: { type: Array, default: [] }, // array of recommendations (youtube/spotify objects)
    createdAt: { type: Date, default: Date.now },
  },
  { _id: false }
);

const UserSchema = new Schema(
  {
    name: { type: String, required: true, trim: true },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },
    passwordHash: { type: String, required: true },
    lastEmotion: { type: String, default: null },
    recommendations: { type: [RecommendationSchema], default: [] },
  },
  { timestamps: true }
);

UserSchema.index({ email: 1 }, { unique: true });

module.exports = model('User', UserSchema);
