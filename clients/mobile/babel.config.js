module.exports = function api(api) {
  api.cache(true);
  return {
    presets: ["babel-preset-expo"],
  };
};
